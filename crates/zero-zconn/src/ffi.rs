//! C-FFI exports for ZeroPlatform (.NET 8/9 C# P/Invoke) integration.

use crate::client::ZConnClient;
use crate::config::ClientConfig;
use zero_input::MouseEvent;

/// Opaque pointer to a ZConn client instance.
#[no_mangle]
pub extern "C" fn zconn_client_create(width: u32, height: u32) -> *mut ZConnClient {
    let client = Box::new(ZConnClient::new(ClientConfig::default(), width, height));
    Box::into_raw(client)
}

/// Frees an allocated ZConn client instance.
///
/// # Safety
/// `client` must be a valid pointer returned by `zconn_client_create`.
#[no_mangle]
pub unsafe extern "C" fn zconn_client_destroy(client: *mut ZConnClient) {
    if !client.is_null() {
        drop(Box::from_raw(client));
    }
}

/// Ingests a raw UDP datagram from the host into the client decoding pipeline.
///
/// # Safety
/// `client` and `data` must be non-null and point to valid memory.
#[no_mangle]
pub unsafe extern "C" fn zconn_client_ingest(
    client: *mut ZConnClient,
    data: *const u8,
    len: usize,
) -> i32 {
    if client.is_null() || data.is_null() || len == 0 {
        return -1;
    }
    let slice = std::slice::from_raw_parts(data, len);
    (*client).handle_incoming_packet(slice);
    0
}

/// Retrieves the raw pointer to the active BGRA frame buffer for Direct3D / WPF presentation.
///
/// # Safety
/// `client`, `out_ptr`, and `out_len` must be valid non-null pointers.
#[no_mangle]
pub unsafe extern "C" fn zconn_client_get_frame(
    client: *mut ZConnClient,
    out_ptr: *mut *const u8,
    out_len: *mut usize,
) -> i32 {
    if client.is_null() || out_ptr.is_null() || out_len.is_null() {
        return -1;
    }
    let surface = (*client).render_surface();
    *out_ptr = surface.as_ptr();
    *out_len = surface.len();
    0
}

/// Synthesizes and serializes a mouse movement packet for dispatch to the remote host.
///
/// # Safety
/// `client` and `out_buf` must be valid pointers with at least `max_len` bytes.
#[no_mangle]
pub unsafe extern "C" fn zconn_client_send_mouse(
    client: *mut ZConnClient,
    norm_x: f32,
    norm_y: f32,
    out_buf: *mut u8,
    max_len: usize,
) -> i32 {
    if client.is_null() || out_buf.is_null() || max_len == 0 {
        return -1;
    }
    let ev = MouseEvent::MoveAbsolute {
        norm_x,
        norm_y,
        display_id: 0,
    };
    let pkt = (*client).build_mouse_packet(&ev, 0);
    if pkt.len() > max_len {
        return -2;
    }
    std::ptr::copy_nonoverlapping(pkt.as_ptr(), out_buf, pkt.len());
    pkt.len() as i32
}

/// Retrieves the local machine's formatted 9-digit device ID (e.g. "912 345 678").
///
/// # Safety
/// `out_buf` must be a valid non-null pointer with at least `max_len` bytes.
#[no_mangle]
pub unsafe extern "C" fn zconn_get_local_device_id(out_buf: *mut u8, max_len: usize) -> i32 {
    if out_buf.is_null() || max_len < 12 {
        return -1;
    }
    let dev = zero_tunnel::DeviceId::generate_local();
    let s = dev.to_formatted_string();
    let bytes = s.as_bytes();
    std::ptr::copy_nonoverlapping(bytes.as_ptr(), out_buf, bytes.len());
    if max_len > bytes.len() {
        *out_buf.add(bytes.len()) = 0; // Null-terminate for C/C# PInvoke
    }
    bytes.len() as i32
}

/// Generates a random 6-character one-time session PIN.
///
/// # Safety
/// `out_buf` must be a valid non-null pointer with at least `max_len` bytes.
#[no_mangle]
pub unsafe extern "C" fn zconn_generate_pin(seed: u64, out_buf: *mut u8, max_len: usize) -> i32 {
    if out_buf.is_null() || max_len < 7 {
        return -1;
    }
    let pin = zero_tunnel::PasswordGenerator::generate_pin(seed);
    let bytes = pin.as_bytes();
    std::ptr::copy_nonoverlapping(bytes.as_ptr(), out_buf, bytes.len());
    if max_len > bytes.len() {
        *out_buf.add(bytes.len()) = 0; // Null-terminate for C/C# PInvoke
    }
    bytes.len() as i32
}
