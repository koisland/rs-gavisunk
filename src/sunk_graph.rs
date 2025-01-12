use distmat::DistMatrix;
use eyre::bail;
use itertools::Itertools;
use petgraph::graph::NodeIndex;
use petgraph::{algo::tarjan_scc, Graph};
use polars::prelude::*;
use rayon::iter::{IntoParallelIterator, ParallelIterator};
use std::collections::{HashMap, HashSet};
use std::ops::Not;

const MIN_READ_LEN: u64 = 10000;

fn get_largest_sunk_graph_component(
    df_grp: &DataFrame,
    rname: &str,
) -> eyre::Result<Option<Vec<i64>>> {
    let cpos_col = df_grp.column("cpos")?;
    let rpos_col = df_grp.column("rpos")?;
    let id_col = df_grp.column("id")?;

    // Calculate pairwise distance between both self contig and read sunk positions
    let cpos_dst_arr = Series::new(
        "cpos_dst".into(),
        DistMatrix::from_pw_distances(cpos_col.i64()?.cont_slice()?)
            .into_inner()
            .1,
    );
    let rpos_dst_arr = Series::new(
        "rpos_dst".into(),
        DistMatrix::from_pw_distances(rpos_col.i64()?.cont_slice()?)
            .into_inner()
            .1,
    );
    // Only take half of entire pairwise mtx.
    /*
        - 1 2 3
        1 0 0 0
        2 1 0 0
        3 1 1 0
    */
    let rpos_sign_arr = Series::new(
        "rpos_sign".into(),
        DistMatrix::from_pw_distances_with(rpos_col.i64()?.cont_slice()?, |a, b| a > b)
            .into_inner()
            .1,
    );

    // Keep track of id to position correspondence.
    let id_comb = id_col
        .i64()?
        .iter()
        .flatten()
        .combinations(2)
        .flat_map(|c| c.into_iter().collect_tuple::<(i64, i64)>())
        .collect_vec();
    let rpos_comb = rpos_col
        .i64()?
        .iter()
        .flatten()
        .combinations(2)
        .flat_map(|c| c.into_iter().collect_tuple::<(i64, i64)>())
        .collect_vec();

    /*
    For each read, a matrix of all pairwise inter-SUNK distances within the read is generated using NumPy
    and compared to expected distances from the assembly,
    allowing ±2% variation in length for a given distance by default
    */
    let pos_diff =
        (rpos_dst_arr.cast(&DataType::Float32)? / cpos_dst_arr.cast(&DataType::Float32)?)?;
    let mask = pos_diff.lt(1.1)? & pos_diff.gt(0.9)?;

    if mask.sum() < Some(1) {
        log::debug!("SUNKs not within 2% variation in length for {rname}");
        return Ok(None);
    }
    let Some(true_orient) = ({
        // Only calculate on masked version
        let df_max_sign = rpos_sign_arr
            .filter(&mask)?
            .value_counts(false, false, "count".into(), false)?
            .lazy()
            .filter(col("count").eq(col("count").max()))
            .select([col("rpos_sign")])
            .first()
            .collect()?;
        df_max_sign.column("rpos_sign")?.bool()?.first()
    }) else {
        bail!("Cannot determine true orient for {rname}.");
    };

    // Generate new mask that checks if is true orientation
    // TODO: Double check.
    let mask_true_orient = if true_orient {
        rpos_sign_arr.bool()? & &mask
    } else {
        rpos_sign_arr.bool()?.not() & mask.clone()
    };

    // Get SUNK and read position with correct orientation.
    let (ids_1, ids_2): (Vec<i64>, Vec<i64>) = id_comb.into_iter().unzip();
    let (pos_1, pos_2): (Vec<i64>, Vec<i64>) = rpos_comb.into_iter().unzip();
    let col_id_1 = Column::new("id_1".into(), ids_1).filter(&mask_true_orient)?;
    let col_id_2 = Column::new("id_2".into(), ids_2).filter(&mask_true_orient)?;
    let col_pos_1 = Column::new("pos_1".into(), pos_1).filter(&mask_true_orient)?;
    let col_pos_2 = Column::new("pos_2".into(), pos_2).filter(&mask_true_orient)?;

    // Find id pair groups with multiple identical sunks.
    // We do this here instead of in polars as would require cloning df twice to perform agg + uniq operation.
    let multi_sunk_grps: HashSet<(i64, i64)> = col_id_1
        .i64()?
        .iter()
        .flatten()
        .zip(col_id_2.i64()?.iter().flatten())
        .zip(
            col_pos_1
                .i64()?
                .iter()
                .flatten()
                .zip(col_pos_2.i64()?.iter().flatten()),
        )
        // Sort and group by pair
        .sorted_by(|(id_pair_1, _), (id_pair_2, _)| id_pair_1.cmp(&id_pair_2))
        .chunk_by(|(id_pair, _)| *id_pair)
        .into_iter()
        // Count number of unique SUNK positions per group.
        // Then mark groups if number of unique SUNK positions greater than 2.
        // ex.
        //  ID ID2 pos1 pos2
        //  1  2   1    3    <- multiple sunk positions
        //  1  2   2    3    <-
        .flat_map(|(grp, grps)| {
            let mut seen_pos = HashSet::new();
            for (_, (pos_1, pos_2)) in grps {
                seen_pos.insert(pos_1);
                seen_pos.insert(pos_2);
            }
            // If greater than 2, indicates that more that one row (multiple sunk positions) for one read id pair
            (seen_pos.len() > 2).then_some(grp)
        })
        .collect();

    let is_multi_sunk = Column::new(
        "is_multi_sunk".into(),
        col_id_1
            .i64()?
            .iter()
            .flatten()
            .zip(col_id_2.i64()?.iter().flatten())
            .map(|(a, b)| multi_sunk_grps.contains(&(a, b)))
            .collect_vec(),
    );

    let cols_subset_id_pos_comb = DataFrame::new(vec![
        col_id_1,
        col_id_2,
        col_pos_1,
        col_pos_2,
        is_multi_sunk,
    ])?
    .lazy()
    // Drop other rows that have dupe sunks.
    .unique_stable(
        Some(vec!["id_1".into(), "id_2".into(), "is_multi_sunk".into()]),
        UniqueKeepStrategy::First,
    )
    .drop([col("is_multi_sunk")])
    .collect()?
    .take_columns();

    let [col_id_1, col_id_2, col_pos_1, col_pos_2] = &cols_subset_id_pos_comb[..] else {
        bail!("Insufficient num of columns.")
    };

    let mut graph: Graph<i64, i64, petgraph::Undirected> = Graph::new_undirected();
    // Add and store nodes
    let node_idxs: HashMap<i64, NodeIndex> = col_id_1
        .i64()?
        .iter()
        .flatten()
        .chain(col_id_2.i64()?.iter().flatten())
        .unique()
        .map(|id| (id, graph.add_node(id)))
        .collect();
    // Add edges.
    for ((id_1, id_2), (pos_1, pos_2)) in col_id_1
        .i64()?
        .iter()
        .flatten()
        .zip(col_id_2.i64()?.iter().flatten())
        .zip(
            col_pos_1
                .i64()?
                .iter()
                .flatten()
                .zip(col_pos_2.i64()?.iter().flatten()),
        )
    {
        let (Some(n1), Some(n2)) = (node_idxs.get(&id_1), node_idxs.get(&id_2)) else {
            unreachable!("ID not added to graph. Node index not found.")
        };
        graph.add_edge(*n1, *n2, (id_2 - id_1) - (pos_2 - pos_1));
    }
    // Use tarjan's algo to find all connected components.
    let components = tarjan_scc(&graph);
    // TODO: Filter components by additional heuristics?
    // See weight above.
    let Some(largest_component) = components.iter().max_by(|a, b| a.len().cmp(&b.len())) else {
        log::debug!("No components found in SUNK graph for {rname}.");
        return Ok(None);
    };

    Ok(Some(
        largest_component
            .iter()
            .flat_map(|node| graph.node_weight(*node))
            .cloned()
            .collect(),
    ))
}

pub fn create_sunk_graph(
    df_read_sunks: &DataFrame,
    read_lens: &HashMap<String, u64>,
    df_bad_sunks: &DataFrame,
) -> eyre::Result<()> {
    let lf_read_sunks = df_read_sunks
        .clone()
        .lazy()
        .with_column((col("ctg") + lit(":") + col("group").cast(DataType::String)).alias("id"))
        .join(
            df_bad_sunks.clone().lazy(),
            [col("id")],
            [col("id")],
            JoinArgs::new(JoinType::Left),
        )
        .with_column(col("group").alias("id"))
        // Filter out bad sunks.
        .filter(col("count").is_null());

    let lf_multisunk = lf_read_sunks
        .clone()
        .group_by([col("read")])
        .agg([col("id").n_unique().alias("id_count")])
        .sort(["id_count"], Default::default())
        .filter(col("id_count").gt(1));

    let lf_sunk_pos = lf_read_sunks
        .join(
            lf_multisunk,
            [col("read")],
            [col("read")],
            JoinArgs::new(JoinType::Left),
        )
        .filter(col("id_count").is_not_null())
        .sort(["read", "rpos"], Default::default())
        .unique(None, UniqueKeepStrategy::First);

    let (col_reads, col_read_len): (Vec<String>, Vec<u64>) = read_lens.clone().into_iter().unzip();

    let df_sunk_pos_w_len = lf_sunk_pos
        .join(
            DataFrame::new(vec![
                Column::new("read".into(), col_reads),
                Column::new("read_length".into(), col_read_len),
            ])?
            .lazy(),
            [col("read")],
            [col("read")],
            JoinArgs::new(JoinType::Left),
        )
        .filter(col("read_length").gt(MIN_READ_LEN))
        .sort(["cpos", "rpos"], Default::default())
        .collect()?;

    let (rnames, ids): (Vec<String>, Vec<i64>) = df_sunk_pos_w_len
        // .lazy()
        // .filter(col("read").eq(lit("00752430-b346-4029-8e6b-5aed0ac718f6")))
        // .collect()?
        .partition_by(["read"], true)?
        .into_par_iter()
        .flat_map(|df_grp| {
            let rname = df_grp
                .column("read")
                .unwrap()
                .str()
                .unwrap()
                .first()
                .unwrap();
            if let Some(ids) = get_largest_sunk_graph_component(&df_grp, rname).unwrap() {
                Some((vec![rname.to_owned(); ids.len()], ids))
            } else {
                None
            }
        })
        .reduce(
            || (vec![], vec![]),
            |(mut r1, mut p1), (mut r2, mut p2)| {
                r1.append(&mut r2);
                p1.append(&mut p2);
                (r1, p1)
            },
        );
    let df_output_sunks = DataFrame::new(vec![
        Column::new("read".into(), rnames),
        Column::new("id".into(), ids),
    ])?;

    println!("{:?}", df_output_sunks);

    Ok(())
}

#[cfg(test)]
mod test {
    use crate::create_sunk_graph;
    use std::{
        collections::HashMap,
        fs::File,
        io::{BufRead, BufReader},
    };

    use itertools::Itertools;
    use polars::prelude::*;

    // #[test]
    // fn test_run() {
    //     let df_read_sunks = {
    //         let mut df = CsvReadOptions::default()
    //             .with_has_header(false)
    //             .with_parse_options(CsvParseOptions::default().with_separator(b'\t'))
    //             .try_into_reader_with_file_path(Some(
    //                 "ignore/test_graph/haplotype1-0000033_hap1.sunkpos".into(),
    //             ))
    //             .unwrap()
    //             .finish()
    //             .unwrap();
    //         df.set_column_names(["read", "rpos", "ctg", "cpos", "group"])
    //             .unwrap();
    //         df
    //     };

    //     let read_lens: HashMap<String, u64> = {
    //         let fh = File::open("ignore/test_graph/hap1.rlen").unwrap();
    //         BufReader::new(fh)
    //             .lines()
    //             .flat_map(|l| {
    //                 let line = l.unwrap();
    //                 let Some((x, y)) = line.trim().split("\t").collect_tuple::<(&str, &str)>()
    //                 else {
    //                     return None;
    //                 };
    //                 Some((x.to_owned(), y.parse::<u64>().unwrap()))
    //             })
    //             .collect()
    //     };
    //     let df_bad_sunks: DataFrame = {
    //         let df = CsvReadOptions::default()
    //             .with_has_header(false)
    //             .with_parse_options(CsvParseOptions::default().with_separator(b'\t'))
    //             .try_into_reader_with_file_path(Some("ignore/test_graph/bad_sunks.txt".into()))
    //             .unwrap()
    //             .finish()
    //             .unwrap();
    //         df.lazy()
    //             .with_column(lit(1).alias("count"))
    //             .rename(["column_1"], ["id"], true)
    //             .collect()
    //             .unwrap()
    //     };

    //     create_sunk_graph(&df_read_sunks, &read_lens, &df_bad_sunks).unwrap();
    // }
}
