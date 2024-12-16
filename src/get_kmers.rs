use core::str;
use std::collections::{HashMap, HashSet};

use super::io::Fasta;
use kmers::{self, Kmer};
use polars::prelude::*;
use rayon::prelude::*;


/// Extract all k-mers counts and starting positions from a given sequence.
/// * Mimic behavior of jellyfish in counting canonical kmers.
///     * `jellyfish -C -m kmer_size`
///     * https://www.genome.umd.edu/docs/JellyfishUserGuide.pdf
///         * 1.1.1 Counting k-mers in sequencing reads
pub fn get_kmer_counts_pos(
    fasta: &str,
    name: &str,
    len: u64,
    kmer_size: usize,
) -> eyre::Result<HashMap<Kmer, (usize, usize)>> {
    let mut fh = Fasta::new(fasta)?;
    let rec = fh.fetch(name, 1, len.try_into()?)?;
    let mut indices: HashMap<Kmer, (usize, usize)> = HashMap::new();
    // Get both fwd and revcomp kmers.
    // Keep track of count and first occurence.
    Kmer::with_many_both_pos(kmer_size, rec.sequence(), |pos, x, y| {
        indices
            .entry(x.clone())
            .and_modify(|(cnt, _)| *cnt += 1)
            .or_insert((1, pos + 1 - kmer_size));
        indices
            .entry(y.clone())
            .and_modify(|(cnt, _)| *cnt += 1)
            .or_insert((1, pos + 1 - kmer_size));
    });
    Ok(indices)
}

/// Get singlely unique kmers in the give fasta file of `kmer_size`.
/// * `fasta` - Fasta file handle.
/// * `kmer_size` - kmer size.
/// * `canonical` - Get canonical kmers (Both fwd + revcomp only count as 1).
pub fn get_sunk_positions(fasta: Fasta, kmer_size: usize, canonical: bool) -> eyre::Result<DataFrame> {
    let all_seq_lens: Vec<(String, u64)> = fasta
        .index
        .as_ref()
        .iter()
        .map(|rec| {
            (
                String::from_utf8(rec.name().to_vec()).unwrap(),
                rec.length(),
            )
        })
        .collect();
    let mut all_kmer_indices: HashMap<String, HashMap<Kmer, (usize, usize)>> = all_seq_lens
        .into_par_iter()
        .map(|(name, len)| {
            let kmer_indices =
                get_kmer_counts_pos(fasta.fname.to_str().unwrap(), &name, len, kmer_size).unwrap();
            (name, kmer_indices)
        })
        .collect();

    // Sum up kmer counts across all sequences.
    let mut kmer_cnts: HashMap<Kmer, usize> =
        all_kmer_indices.values().fold(HashMap::new(), |mut a, b| {
            for (kmer, (cnt, _)) in b.iter() {
                *a.entry(kmer.clone()).or_default() += *cnt
            }
            a
        });
    // Only get SUNKs.
    kmer_cnts.retain(|_, cnt| *cnt == 1);

    all_kmer_indices.par_iter_mut().for_each(|(_, kmers)| {
        // Get kmers that only occur once.
        kmers.retain(|k, _| kmer_cnts.contains_key(k));

        if canonical {
            // TODO: There's probably a better way to do this.
            let mut keep_list: HashSet<Kmer> = HashSet::new();
            let mut remove_list: HashSet<Kmer> = HashSet::new();
            for (kmer, _) in kmers.iter() {
                // Does this kmer have a revcomp version in the kmers map?
                // If so, remove it to avoid double counting.
                let rev_comp_kmer = kmer.rev_comp(kmer_size);
                if !remove_list.contains(&rev_comp_kmer)
                    && kmers.contains_key(&rev_comp_kmer)
                    && !keep_list.contains(&rev_comp_kmer)
                {
                    keep_list.insert(kmer.clone());
                    remove_list.insert(rev_comp_kmer);
                }
            }
            // Remove kmers that have a revcomp.
            kmers.retain(|k, _| keep_list.contains(k));
        }
    });

    let mut seqs = vec![];
    let mut kmers = vec![];
    let mut starts = vec![];
    for (name, kmer_cnts) in all_kmer_indices {
        for (kmer, (_, pos)) in kmer_cnts {
            seqs.push(name.clone());
            kmers.push(kmer.render(kmer_size));
            starts.push(pos as u64);
        }
    }
    let df_sunks: DataFrame = DataFrame::new(vec![
        Column::new("name".into(), seqs),
        Column::new("start".into(), starts),
        Column::new("kmer".into(), kmers),
    ])?;

    df_sunks
        .lazy()
        .sort(["name", "start"], Default::default())
        .with_column(
            // Calculate run-length encoding to group values.
            (col("start") - col("start").shift_and_fill(lit(1), lit(0)))
                .rle_id()
                .over(["name"])
                .alias("group"),
        )
        .with_column(
            // grp:     0  0  0 1 2
            // pos:     1 ... 5 9 10
            // grp_new: 1  1  1 3 3
            // Check if group num is 0 due to shift.
            when(col("group").eq(lit(0)))
                .then(lit(1))
                // Check if group num is even.
                .when((col("group") % lit(2)).eq(lit(0)))
                // Even values are on edge of position transition
                .then(col("group") + lit(1))
                .otherwise(col("group"))
                .over(["name"])
                .alias("group"),
        )
        .with_column(col("start").first().over(["name", "group"]).alias("group"))
        .collect()
        .map_err(|err| eyre::ErrReport::msg(err))
}
