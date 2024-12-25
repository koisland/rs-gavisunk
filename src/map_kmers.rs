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
    let seq = rec.sequence();
    let mut mtches = vec![];

    let Some(kmer_size) = sunks.first().map(|k| k.len()) else {
        bail!("No SUNKs given.")
    };

    let mut idx = SimplePosIndex::new(kmer_size);
    idx.add_seq_both(seq);

    for sunk in sunks {
        for pos in idx.find(&Kmer::make(&sunk).unwrap()) {
            mtches.push((ctg, *sunk, *pos));
        }
    }
    Ok(mtches)
}

pub fn map_sunks_to_seqs(fa: Fasta, df_sunks: &DataFrame) -> eyre::Result<DataFrame> {
    let lengths = fa.lengths();
    log::info!("Total number of reads: {}", lengths.len());

    let col_sunks = df_sunks.column("kmer")?;
    let sunks: Vec<&str> = col_sunks.str()?.into_iter().flatten().collect();
    log::info!("Total number of SUNKs to map: {}", sunks.len());

    let mapped_sunks: Vec<(&str, &str, usize)> = lengths
        .par_iter()
        .map(|(seq, len)| map_sunks_to_seq(&sunks, &fa.fname, seq, 1, *len as u32).unwrap())
        .reduce(Vec::new, |mut a, b| {
            a.extend(b);
            a
        })
        .into_iter()
        .collect();


    let mut seqs = vec![];
    let mut kmers = vec![];
    let mut positions = vec![];
    for (seq, kmer, pos) in mapped_sunks.into_iter() {
        seqs.push(seq);
        kmers.push(kmer);
        positions.push(pos as u64);
    }

    let df_final = DataFrame::new(vec![
        Column::new("seq".into(), seqs),
        Column::new("kmer".into(), kmers),
        Column::new("pos".into(), positions),
    ])?
    .join(df_sunks, ["kmer"], ["kmer"], JoinArgs::new(JoinType::Left))?
    .lazy()
    .group_by([col("seq"), col("name"), col("group")])
    .agg([
        col("start").first(),
        col("pos").sort_by(["start"], Default::default()).first()
    ])
    .select([
        col("seq"),
        col("pos"),
        col("name"),
        col("start"),
        col("group"),
    ])
    .sort(["seq"], Default::default())
    .collect()?;

    log::info!("Total SUNKs mapped: {}", df_final.shape().0);

    Ok(df_final)
}
