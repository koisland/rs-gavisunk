use polars::prelude::*;

const DEFAULT_BANDWIDTH: u64 = 2500;
const DEFAULT_GOOD_SUNK_THR: u64 = 1;

/// Determine which read best matches a given contig based on mapped SUNK position and determine its orientation.
///
/// * From Bioinformatics paper:
///
/// > Each ONT read is assigned to its best-matching HiFi-assembled contig and
/// > orientation by comparing the locations of read SUNKs to assembly SUNKs
/// > within a diagonal band centered on the median SUNK location for that read
///
/// # Arguments
/// * `df_read_sunk_pos`
///     * [`DataFrame`] of read SUNK positions with columns: `[read, rpos, chrom, cpos]`
/// * `bandwidth`
///     * Number of bps around median SUNK position to use in filtering SUNKs.
///     * **A 'good' SUNK is one within this bandwidth.**
/// * `good_sunk_threshold`
///     * Number of 'good' SUNKs required to not filter read.
///
/// # Returns
/// * [`DataFrame`] of reads assigned to contigs and their orientation.
///     * Has columns: `[read, chrom, sunks_within_bandwidth, ort]`
pub fn assign_read_to_ctg_w_ort(
    df_read_sunk_pos: &DataFrame,
    bandwidth: Option<u64>,
    good_sunk_threshold: Option<u64>,
) -> eyre::Result<DataFrame> {
    let bandwidth = bandwidth.unwrap_or(DEFAULT_BANDWIDTH);
    let good_sunk_threshold = good_sunk_threshold.unwrap_or(DEFAULT_GOOD_SUNK_THR);
    let lf_read_sunk_pos = df_read_sunk_pos.clone().lazy();

    log::info!("Using median SUNK bandwidth: {bandwidth}");
    log::info!("Requiring a read to have at least {good_sunk_threshold} SUNK(s) within bandwidth.");

    let lf_ort = df_read_sunk_pos
        .select(["read", "ctg", "cpos", "rpos"])?
        .lazy()
        // Filter reads with only sunk over read and chrom.
        .filter(col("read").len().over(["read", "ctg"]).gt(lit(1)))
        .group_by(["read", "ctg"])
        .agg([
            // Calculate a 1D gradient to get direction of sunks in chrom.
            // ex. [1, 3, 6, 7] => [ 2,  3,  1] => mean is 2 => + (ascending)
            // ex. [9, 7, 3, 1] => [ -2, -4, -2] => mean is -2.67  => - (descending)
            //
            // Similar to np.gradient but without resulting same length vec.
            // * See here for more https://stackoverflow.com/a/24633888
            (col("cpos") - col("cpos").shift(lit(1)))
                .drop_nulls()
                .mean()
                .gt(lit(0))
                .alias("cort"),
            (col("rpos") - col("rpos").shift(lit(1)))
                .drop_nulls()
                .mean()
                .gt(lit(0))
                .alias("rort"),
        ])
        .with_column(
            // Final read ort
            // Both directions equal, read is + ort. Otherwise, - ort.
            when(col("rort").and(col("cort")))
                .then(lit("+"))
                .otherwise(lit("-"))
                .alias("ort"),
        );

    let df = lf_read_sunk_pos
        // Filter reads with only sunk over read and chrom.
        .filter(col("read").len().over(["read", "ctg"]).gt(lit(1)))
        // Add orientation.
        .join(
            lf_ort.lazy(),
            [col("read"), col("ctg")],
            [col("read"), col("ctg")],
            JoinArgs::new(JoinType::Left),
        )
        // Calculate adjusted start position of SUNK based on orientation.
        // n = sunk start position
        // >/< = coord bounds and orientation.
        // n* = adjusted start position of SUNK within ctg
        //                 2*
        // fwd: (read)  >    4 >
        //      (ctg)   >      6   >
        //                       7*
        // rev: (read)  < 1   <
        //      (ctg)   >      6   >
        .with_column(
            when(col("ort").eq(lit("+")))
                .then(col("cpos") - col("rpos"))
                .otherwise(col("cpos") + col("rpos"))
                .alias("apos"),
        )
        // Then count sunks valid sunks around median position within bandwidth.
        .with_column(
            (col("apos") - col("apos").median())
                .abs()
                .lt(bandwidth)
                .sum()
                .over(["read", "ctg"])
                .alias("sunks_within_bandwidth"),
        )
        // Choose based on maximum number of sunks within bandwidth.
        .filter(
            col("sunks_within_bandwidth")
                .eq(col("sunks_within_bandwidth").max())
                .over(["read"]),
        )
        .group_by(["read"])
        // Resolve ties by taking just first row.
        .agg([all().first()])
        // Filter reads with only n good sunks
        .filter(col("sunks_within_bandwidth").gt(good_sunk_threshold))
        .select([
            col("read"),
            col("ctg"),
            col("sunks_within_bandwidth"),
            col("ort"),
        ])
        .collect()?;

    log::info!("Total number of valid reads: {}", df.shape().0);
    Ok(df)
}
