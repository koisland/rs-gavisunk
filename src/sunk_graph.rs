use std::collections::HashMap;

use distmat::DistMatrix;
use itertools::Itertools;
use polars::prelude::*;

const MIN_READ_LEN: u64 = 10000;

pub fn create_sunk_graph(
    df_asm_sunks: &DataFrame,
    asm_lens: &HashMap<String, u64>,
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
        .collect()?;

    // dbg!(&df_sunk_pos_w_len);

    for df_grp in df_sunk_pos_w_len.partition_by(["read"], true)? {
        let cpos_col = df_grp.column("cpos")?;
        let rpos_col = df_grp.column("rpos")?;
        let id_col = df_grp.column("id")?;

        let cpos_dst_mtx = DistMatrix::from_pw_distances(cpos_col.i64()?.cont_slice()?);
        let rpos_dst_mtx = DistMatrix::from_pw_distances(rpos_col.i64()?.cont_slice()?);
        let rpos_sign_dst_mtx =
            DistMatrix::from_pw_distances_with(rpos_col.i64()?.cont_slice()?, |a, b| a > b);

        // Keep track of id to position correspondence.
        let id_comb = id_col
            .str()?
            .as_string()
            .iter()
            .flatten()
            .combinations(2)
            .collect_vec();
        let rpos_comb = cpos_col
            .i64()?
            .iter()
            .flatten()
            .combinations(2)
            .collect_vec();

        // Only store half of matrix.
        let mut mask = Vec::with_capacity(cpos_dst_mtx.size() / 2);
        let mut num_passing = 0;
        for (c_row, p_row) in cpos_dst_mtx.iter_rows().zip(rpos_dst_mtx.iter_rows()) {
            for (c_dst, p_dst) in c_row.into_iter().zip(p_row.into_iter()) {
                let dst_prop = p_dst as f32 / c_dst as f32;
                let passing_kmer = (dst_prop < 1.1) & (dst_prop > 0.9);
                num_passing += usize::from(passing_kmer);
                mask.push(passing_kmer);
            }
        }
        if num_passing < 1 {
            continue;
        }

        dbg!(df_grp);
        break;
    }
    Ok(())
}
