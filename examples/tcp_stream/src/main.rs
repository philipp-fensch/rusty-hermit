use std::net::TcpStream;
use std::io::{Write, Read};

#[cfg(target_os = "hermit")]
extern crate hermit_sys;

fn main() -> std::io::Result<()> {
    let mut stream = TcpStream::connect("10.0.5.1:1806")?;
    const NUM: i32 = 10;

    println!("[Client] Write to Stream: {}", NUM);
    stream.write(&i32::to_be_bytes(NUM))?;

    let mut buf = [0; 4];
    stream.read(&mut buf)?;
    let integer = i32::from_be_bytes(buf); 
    println!("[Client] Received {}: {} * {} = {}", integer, NUM, NUM, integer);

    assert!(NUM * NUM == integer, "[Client] Result wrong: Expected {}, got {}", NUM * NUM, integer);
    Ok(())
}
