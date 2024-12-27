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
///
/// # Arguments
/// * `fasta`
///     * Fasta file handle
/// * `name`
///     * Name of sequence.
/// * `len`
///     * Length of sequence.
/// * `kmer_size`
///     * kmer size.
///
/// # Returns
/// * Map of kmers with the their count and first encountered position.
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
            .or_insert((1, pos + 1));
        indices
            .entry(y.clone())
            .and_modify(|(cnt, _)| *cnt += 1)
            .or_insert((1, pos + 1));
    });
    Ok(indices)
}

/// Get singlely unique kmers in the give fasta file of `kmer_size`.
///
/// # Arguments
/// * `fasta`
///     * Fasta file handle.
/// * `kmer_size`
///     * kmer size.
/// * `canonical`
///     * Get canonical kmers (Both fwd + revcomp only count as 1).
///
/// # Returns
/// * [`DataFrame`] of SUNK positions with columns `[name, start, kmer, group]`.
pub fn get_sunk_positions(
    fasta: Fasta,
    kmer_size: usize,
    canonical: bool,
) -> eyre::Result<DataFrame> {
    let all_seq_lens: Vec<(String, u64)> = fasta.lengths();
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

    let mut ctgs = vec![];
    let mut kmers = vec![];
    let mut positions = vec![];
    for (name, kmer_cnts) in all_kmer_indices {
        for (kmer, (_, pos)) in kmer_cnts {
            ctgs.push(name.clone());
            kmers.push(kmer.render(kmer_size));
            positions.push(pos as u64);
        }
    }
    let df_sunks: DataFrame = DataFrame::new(vec![
        Column::new("ctg".into(), ctgs),
        Column::new("cpos".into(), positions),
        Column::new("kmer".into(), kmers),
    ])?;

    let df_sunks_final = df_sunks
        .lazy()
        .sort(["ctg", "cpos"], Default::default())
        .with_column(
            // Calculate run-length encoding to group values.
            (col("cpos") - col("cpos").shift_and_fill(lit(1), lit(0)))
                .gt(lit(1))
                .rle_id()
                .over(["ctg"])
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
                .over(["ctg"])
                .alias("group"),
        )
        // Set group number to be the first position in adjacent sunks.
        .with_column(col("cpos").first().over(["ctg", "group"]).alias("group"))
        .collect()?;

    log::info!("Total number of SUNKs: {}", df_sunks_final.shape().0);
    Ok(df_sunks_final)
}
