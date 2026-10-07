//! FastCGI request packaging and response extraction for PHP-FPM communication.

use crate::record::{FcgiHeader, FcgiNameValuePair, FcgiRecordType, FCGI_VERSION_1};
use zero_core::error::{ZeroError, ZeroResult};

/// Helper to serialize a complete FastCGI HTTP request stream into a buffer.
pub struct FastCgiRequestBuilder;

impl FastCgiRequestBuilder {
    /// Builds a minimal FastCGI probe/request packet to invoke a PHP script on PHP-FPM.
    ///
    /// Writes:
    /// 1. `FCGI_BEGIN_REQUEST` (role = 1 Responder, keep_conn = 1) -> 8 + 8 = 16 bytes
    /// 2. `FCGI_PARAMS` with script and environment details
    /// 3. Empty `FCGI_PARAMS` (0 bytes) signaling parameter EOF
    /// 4. Empty `FCGI_STDIN` (0 bytes) signaling stdin EOF
    pub fn build_get_request(
        request_id: u16,
        script_path: &[u8],
        doc_root: &[u8],
        out_buf: &mut [u8],
    ) -> ZeroResult<usize> {
        let mut offset = 0;

        // 1. FCGI_BEGIN_REQUEST Header (8 bytes) + Body (8 bytes)
        let begin_hdr = FcgiHeader {
            version: FCGI_VERSION_1,
            record_type: FcgiRecordType::BeginRequest,
            request_id,
            content_length: 8,
            padding_length: 0,
        };
        begin_hdr.serialize(&mut out_buf[offset..offset + 8])?;
        offset += 8;

        // Role = 1 (Responder, big-endian u16 = [0, 1]), flags = 1 (keep_conn), reserved 5 bytes
        out_buf[offset..offset + 8]
            .copy_from_slice(&[0x00, 0x01, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00]);
        offset += 8;

        // 2. FCGI_PARAMS payload
        let params_content_start = offset + 8;
        let mut params_len = 0;

        // SCRIPT_FILENAME
        params_len += FcgiNameValuePair::encode(
            b"SCRIPT_FILENAME",
            script_path,
            &mut out_buf[params_content_start + params_len..],
        )?;

        // DOCUMENT_ROOT
        params_len += FcgiNameValuePair::encode(
            b"DOCUMENT_ROOT",
            doc_root,
            &mut out_buf[params_content_start + params_len..],
        )?;

        // REQUEST_METHOD = GET
        params_len += FcgiNameValuePair::encode(
            b"REQUEST_METHOD",
            b"GET",
            &mut out_buf[params_content_start + params_len..],
        )?;

        // SERVER_SOFTWARE = ZPanl
        params_len += FcgiNameValuePair::encode(
            b"SERVER_SOFTWARE",
            b"ZPanl/0.1.0",
            &mut out_buf[params_content_start + params_len..],
        )?;

        // Write FCGI_PARAMS Header
        let params_hdr = FcgiHeader {
            version: FCGI_VERSION_1,
            record_type: FcgiRecordType::Params,
            request_id,
            content_length: params_len as u16,
            padding_length: 0,
        };
        params_hdr.serialize(&mut out_buf[offset..offset + 8])?;
        offset += 8 + params_len;

        // 3. Empty FCGI_PARAMS Header (EOF of parameters)
        let empty_params_hdr = FcgiHeader {
            version: FCGI_VERSION_1,
            record_type: FcgiRecordType::Params,
            request_id,
            content_length: 0,
            padding_length: 0,
        };
        empty_params_hdr.serialize(&mut out_buf[offset..offset + 8])?;
        offset += 8;

        // 4. Empty FCGI_STDIN Header (EOF of stdin)
        let empty_stdin_hdr = FcgiHeader {
            version: FCGI_VERSION_1,
            record_type: FcgiRecordType::Stdin,
            request_id,
            content_length: 0,
            padding_length: 0,
        };
        empty_stdin_hdr.serialize(&mut out_buf[offset..offset + 8])?;
        offset += 8;

        Ok(offset)
    }

    /// Extracts STDOUT payload from incoming FastCGI response stream.
    ///
    /// Scans buffer for `FCGI_STDOUT` records and copies response bytes to `dest`.
    /// Returns total response payload bytes extracted.
    pub fn parse_stdout(stream: &[u8], dest: &mut [u8]) -> ZeroResult<usize> {
        let mut stream_offset = 0;
        let mut dest_offset = 0;

        while stream_offset + 8 <= stream.len() {
            let hdr = FcgiHeader::deserialize(&stream[stream_offset..stream_offset + 8])?;
            let record_len = 8 + hdr.content_length as usize + hdr.padding_length as usize;

            if stream_offset + record_len > stream.len() {
                return Err(ZeroError::UnexpectedEndOfBuffer);
            }

            if hdr.record_type == FcgiRecordType::Stdout && hdr.content_length > 0 {
                let payload =
                    &stream[stream_offset + 8..stream_offset + 8 + hdr.content_length as usize];
                if dest_offset + payload.len() > dest.len() {
                    return Err(ZeroError::BufferOverflow);
                }
                dest[dest_offset..dest_offset + payload.len()].copy_from_slice(payload);
                dest_offset += payload.len();
            }

            if hdr.record_type == FcgiRecordType::EndRequest {
                break;
            }

            stream_offset += record_len;
        }

        Ok(dest_offset)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_build_get_request_roundtrip() {
        let mut buf = [0u8; 512];
        let written = FastCgiRequestBuilder::build_get_request(
            1,
            b"/var/www/info.php",
            b"/var/www",
            &mut buf,
        )
        .unwrap();

        assert!(written > 40);

        // First header must be BeginRequest
        let h0 = FcgiHeader::deserialize(&buf[0..8]).unwrap();
        assert_eq!(h0.record_type, FcgiRecordType::BeginRequest);
        assert_eq!(h0.request_id, 1);
    }
}
