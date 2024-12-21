use memchr::memmem;
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
    for sunk in sunks {
        for mtch in memmem::find_iter(seq.as_ref(), sunk) {
            mtches.push((ctg, *sunk, mtch));
        }
    }
    Ok(mtches)
}

pub fn map_sunks_to_seqs(fa: Fasta, df_sunks: &DataFrame) -> eyre::Result<DataFrame> {
    let lengths = fa.lengths();
    log::info!("Total number of reads: {}", lengths.len());

    let col_sunks = df_sunks.column("kmer")?.unique()?;
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

    log::info!("Total SUNKs mapped: {}", mapped_sunks.len());

    let mut ctgs = vec![];
    let mut kmers = vec![];
    let mut positions = vec![];
    for (ctg, kmer, pos) in mapped_sunks.into_iter() {
        ctgs.push(ctg);
        kmers.push(kmer);
        positions.push(pos as u64);
    }

    DataFrame::new(vec![
        Column::new("ctg".into(), ctgs),
        Column::new("kmer".into(), kmers),
        Column::new("pos".into(), positions),
    ])
    .map_err(|err| eyre::ErrReport::msg(err))
}
