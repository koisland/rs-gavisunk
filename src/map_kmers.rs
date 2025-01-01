use eyre::bail;
use kmers::{self, Kmer, SimplePosIndex};
use std::path::PathBuf;

use crate::io::Fasta;
use polars::prelude::*;
use rayon::iter::{IntoParallelRefIterator, ParallelIterator};

fn map_sunks_to_seq<'a, 'b>(
    sunks: &[&'a str],
    fname: &PathBuf,
    ctg: &'b str,
    start: u32,
    end: u32,
) -> eyre::Result<Vec<(&'b str, &'a str, usize)>> {
    let mut fasta = Fasta::new(fname)?;
    let rec = fasta.fetch(ctg, start, end)?;

    let Some(kmer_size) = sunks.first().map(|k| k.len()) else {
        bail!("No SUNKs given.")
    };

    // Use kmer's simple positional index to generate all kmer position indices first.
    // Add both fwd and reverse comp kmers.
    let mut idx = SimplePosIndex::new(kmer_size);
    idx.add_seq_both(rec.sequence());

    // Then iterate thru all sunks and get their 1-based positions within the index.
    Ok(sunks
        .iter()
        .flat_map(|sunk| {
            idx.find(&Kmer::make(sunk).unwrap())
                .iter()
                .map(|pos| (ctg, *sunk, *pos + 1))
        })
        .collect())
}

/// Map sunks from an assembly to reads.
///
/// # Arguments
/// * `fa`
///     * Fasta file handle for reads.
/// * `df_sunks`
///     * [`DataFrame`] with columns `[name, kmer, start, group]`
///
/// # Returns
/// * [`DataFrame`] of SUNKs within reads from the assembly.
///     * With columns `[seq, pos, name, start, group]`
pub fn map_sunks_to_reads(fa: Fasta, df_sunks: &DataFrame) -> eyre::Result<DataFrame> {
    let lengths = fa.lengths();
    log::info!("Found {} reads.", lengths.len());

    let col_sunks = df_sunks.column("kmer")?;
    let sunks: Vec<&str> = col_sunks.str()?.into_iter().flatten().collect();

    let mapped_sunks: Vec<(&str, &str, usize)> = lengths
        .par_iter()
        .map(|(seq, len)| map_sunks_to_seq(&sunks, &fa.fname, seq, 1, *len as u32).unwrap())
        .reduce(Vec::new, |mut a, b| {
            a.extend(b);
            a
        })
        .into_iter()
        .collect();

    let mut reads = Vec::with_capacity(mapped_sunks.len());
    let mut kmers = Vec::with_capacity(mapped_sunks.len());
    let mut positions = Vec::with_capacity(mapped_sunks.len());
    for (read, kmer, pos) in mapped_sunks.into_iter() {
        reads.push(read);
        kmers.push(kmer);
        positions.push(pos as u64);
    }

    let df_final = DataFrame::new(vec![
        Column::new("read".into(), reads),
        Column::new("kmer".into(), kmers),
        Column::new("rpos".into(), positions),
    ])?
    .join(df_sunks, ["kmer"], ["kmer"], JoinArgs::new(JoinType::Left))?
    .lazy()
    .group_by([col("read"), col("ctg"), col("group")])
    .agg([
        col("cpos").first(),
        col("rpos").sort_by(["cpos"], Default::default()).first(),
    ])
    .select([
        col("read"),
        col("rpos"),
        col("ctg"),
        col("cpos"),
        col("group"),
    ])
    .sort(["read", "rpos"], Default::default())
    .collect()?;

    log::info!("Total SUNKs mapped: {}", df_final.shape().0);

    Ok(df_final)
}

pub fn get_good_read_sunks(
    df_read_sunks: &DataFrame,
    df_best_reads_asm: &DataFrame,
) -> eyre::Result<DataFrame> {
    Ok(df_read_sunks
        .inner_join(df_best_reads_asm, ["read", "ctg"], ["read", "ctg"])?
        .select(["read", "rpos", "ctg", "cpos", "group"])?)
}
