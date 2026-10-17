//! Bridge C-order f64 NPY files and MessagePack rows for native Nu matrices.
use npyz::WriterBuilder;
use std::{
    error::Error,
    fs::File,
    io::{self, BufReader, BufWriter, Write},
};

fn main() -> Result<(), Box<dyn Error>> {
    let args: Vec<_> = std::env::args().collect();
    if args.len() != 3 {
        return Err("usage: npy-codec decode|encode PATH".into());
    }
    match args[1].as_str() {
        "decode" => {
            let reader = npyz::NpyFile::new(BufReader::new(File::open(&args[2])?))?;
            if reader.order() != npyz::Order::C || reader.shape().len() != 2 {
                return Err("expected a rank-2 C-order f64 NPY file".into());
            }
            let rows = usize::try_from(reader.shape()[0])?;
            let cols = usize::try_from(reader.shape()[1])?;
            let count = rows.checked_mul(cols).ok_or("shape overflow")?;
            if count == 0 {
                return Err("empty matrices are unsupported".into());
            }
            let mut data = reader.data::<f64>()?;
            let mut output = BufWriter::new(io::stdout().lock());
            // Stream standard MessagePack array headers and f64 values; no row copies.
            write_array(&mut output, rows)?;
            for _ in 0..rows {
                write_array(&mut output, cols)?;
                for _ in 0..cols {
                    let value = data.next().ok_or("truncated NPY data")??;
                    output.write_all(&[0xcb])?;
                    output.write_all(&value.to_be_bytes())?;
                }
            }
            output.flush()?;
        }
        "encode" => {
            let rows: Vec<Vec<f64>> = rmp_serde::from_read(io::stdin().lock())?;
            let cols = rows.first().ok_or("empty matrix")?.len();
            if cols == 0 || rows.iter().any(|row| row.len() != cols) {
                return Err("expected nonempty rectangular f64 rows".into());
            }
            let shape = [rows.len() as u64, cols as u64];
            let output = BufWriter::new(File::create_new(&args[2])?);
            let mut writer = npyz::WriteOptions::new()
                .default_dtype()
                .shape(&shape)
                .writer(output)
                .begin_nd()?;
            for row in rows {
                writer.extend(row)?;
            }
            writer.finish()?;
        }
        _ => return Err("expected decode or encode".into()),
    }
    Ok(())
}

fn write_array(out: &mut impl Write, count: usize) -> io::Result<()> {
    let count = u32::try_from(count).map_err(io::Error::other)?;
    out.write_all(&[0xdd])?;
    out.write_all(&count.to_be_bytes())
}
