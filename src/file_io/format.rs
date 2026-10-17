use crate::gpu::Result;
use cubecl::{CubeElement, ir::ElemType};
use std::fs::File;
use std::io::{Read, Seek, SeekFrom};

pub(super) struct Layout {
    pub shape: Vec<usize>,
    pub dtype: ElemType,
    pub offset: usize,
    pub bytes: usize,
}

pub(super) fn read(file: &mut File) -> Result<Layout> {
    let mut prefix = [0u8; 12];
    file.read_exact(&mut prefix)?;
    if &prefix[..6] != b"\x93NUMPY" {
        return Err("expected a NumPy .npy file".into());
    }
    let (start, len): (usize, usize) = match (prefix[6], prefix[7]) {
        (1, 0) => (
            10,
            usize::from(u16::from_le_bytes(prefix[8..10].try_into()?)),
        ),
        (2 | 3, 0) => (
            12,
            usize::try_from(u32::from_le_bytes(prefix[8..12].try_into()?))?,
        ),
        _ => return Err("unsupported .npy version".into()),
    };
    let total = start.checked_add(len).ok_or("header size overflow")?;
    if total > 65536 {
        return Err(".npy header exceeds 64 KiB".into());
    }
    file.seek(SeekFrom::Start(0))?;
    let mut encoded = vec![0; total];
    file.read_exact(&mut encoded)?;
    // npyz computes strides with unchecked multiplication. Validate dimensions first.
    let literal: py_literal::Value = std::str::from_utf8(&encoded[start..])?.trim().parse()?;
    let dict = literal.as_dict().ok_or("expected NPY header dictionary")?;
    let mut keys = std::collections::BTreeSet::new();
    for (key, value) in dict {
        let key = key.as_string().ok_or("non-string NPY key")?;
        if !keys.insert(key) {
            return Err("duplicate NPY header key".into());
        }
        if key == "descr" && !value.as_string().is_some_and(|s| s == "<f4" || s == "<f8") {
            return Err("expected little-endian f32 or f64".into());
        }
    }
    let dims = dict
        .iter()
        .find(|(key, _)| key.as_string().is_some_and(|s| s == "shape"))
        .and_then(|(_, value)| value.as_tuple())
        .ok_or("expected shape tuple")?;
    if !(1..=2).contains(&dims.len()) {
        return Err("expected rank one or two".into());
    }
    dims.iter().try_fold(8usize, |size, dim| {
        let n: usize = dim
            .as_integer()
            .ok_or("invalid dimension")?
            .to_string()
            .parse()
            .map_err(|_| "invalid dimension")?;
        size.checked_mul(n).ok_or("array size overflow")
    })?;
    file.rewind()?;
    let header = npyz::NpyHeader::from_reader((&mut *file).take(total as u64))?;
    if header.order() != npyz::Order::C || !(1..=2).contains(&header.shape().len()) {
        return Err("expected a C-order one- or two-dimensional array".into());
    }
    let (dtype, width): (_, usize) = if header.dtype() == npyz::DType::new_scalar("<f8".parse()?) {
        (f64::cube_type(), 8)
    } else if header.dtype() == npyz::DType::new_scalar("<f4".parse()?) {
        (f32::cube_type(), 4)
    } else {
        return Err("expected little-endian f32 or f64 .npy data".into());
    };
    let shape = header
        .shape()
        .iter()
        .map(|&n| usize::try_from(n))
        .collect::<std::result::Result<Vec<_>, _>>()?;
    let bytes = shape
        .iter()
        .try_fold(width, |n, &dim| n.checked_mul(dim))
        .ok_or("array size overflow")?;
    if bytes == 0
        || total.checked_add(bytes).ok_or("array size overflow")? as u64 > file.metadata()?.len()
    {
        return Err("empty or truncated .npy array".into());
    }
    Ok(Layout {
        shape,
        dtype,
        offset: total,
        bytes,
    })
}

pub(super) fn header(shape: &[usize], dtype: ElemType, alignment: usize) -> Result<Vec<u8>> {
    if !(1..=2).contains(&shape.len()) || shape.contains(&0) {
        return Err("expected nonempty rank one or two".into());
    }
    if alignment == 0 || !alignment.is_power_of_two() || alignment > 65536 {
        return Err("unsupported import alignment".into());
    }
    let descr = if dtype == f64::cube_type() {
        "<f8"
    } else if dtype == f32::cube_type() {
        "<f4"
    } else {
        return Err("expected f32 or f64".into());
    };
    let dims = shape
        .iter()
        .map(usize::to_string)
        .collect::<Vec<_>>()
        .join(", ");
    let dict = format!("{{'descr': '{descr}', 'fortran_order': False, 'shape': ({dims},), }}");
    let total = dict
        .len()
        .checked_add(13)
        .and_then(|n| n.checked_next_multiple_of(alignment.max(64)))
        .ok_or("header size overflow")?;
    let len = u32::try_from(total - 12)?;
    let mut out = b"\x93NUMPY\x02\x00".to_vec();
    out.extend_from_slice(&len.to_le_bytes());
    out.extend_from_slice(dict.as_bytes());
    out.resize(total - 1, b' ');
    out.push(b'\n');
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    #[test]
    fn rejects_unsupported_dtype_and_fortran_order() {
        let original = header(&[1], f64::cube_type(), 64).unwrap();
        for (from, to, reason) in [
            ("<f8", "<i8", "little-endian"),
            ("False", "True ", "C-order"),
        ] {
            let mut file = tempfile::tempfile().unwrap();
            let dict = std::str::from_utf8(&original[12..])
                .unwrap()
                .replace(from, to);
            assert_eq!(dict.len(), original.len() - 12);
            file.write_all(&original[..12]).unwrap();
            file.write_all(dict.as_bytes()).unwrap();
            file.write_all(&0f64.to_le_bytes()).unwrap();
            file.rewind().unwrap();
            assert!(read(&mut file).err().unwrap().to_string().contains(reason));
        }
    }
    #[test]
    fn rejects_oversized_or_duplicate_shape_before_npyz() {
        for dict in [
            "{'descr':'<f8','fortran_order':False,'shape':(18446744073709551615,2)}",
            "{'descr':'<f8','fortran_order':False,'shape':(1,),'shape':(18446744073709551615,2)}",
        ] {
            let mut file = tempfile::tempfile().unwrap();
            file.write_all(b"\x93NUMPY\x02\x00").unwrap();
            file.write_all(&(dict.len() as u32).to_le_bytes()).unwrap();
            file.write_all(dict.as_bytes()).unwrap();
            file.rewind().unwrap();
            assert!(read(&mut file).is_err());
        }
    }
    #[test]
    fn interoperable_padded_header_and_truncation() {
        let mut file = tempfile::tempfile().unwrap();
        let header = header(&[2, 3], f64::cube_type(), 4096).unwrap();
        assert_eq!(header.len(), 4096);
        file.write_all(&header).unwrap();
        file.set_len(4096 + 48).unwrap();
        file.rewind().unwrap();
        let layout = read(&mut file).unwrap();
        assert_eq!(layout.shape, [2, 3]);
        assert_eq!(layout.bytes, 48);
        file.set_len(4096 + 47).unwrap();
        file.rewind().unwrap();
        assert!(read(&mut file).is_err());
    }
}
