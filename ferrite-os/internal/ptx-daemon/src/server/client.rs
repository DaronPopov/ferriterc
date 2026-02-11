use std::io::{self, Read, Write};
use std::os::unix::net::UnixStream;
use std::thread;
use std::time::Duration;

pub fn connect_and_send(socket: &str, command: &str) -> io::Result<()> {
    let mut stream = UnixStream::connect(socket)?;
    stream.write_all(command.as_bytes())?;
    stream.write_all(b"\n")?;
    stream.flush()?;

    let mut resp = String::new();
    stream.read_to_string(&mut resp)?;
    print!("{}", resp);
    Ok(())
}

pub fn run_watch_client(socket: &str, watch_ms: u64) -> io::Result<()> {
    let is_tty = unsafe { libc::isatty(libc::STDOUT_FILENO) == 1 };

    loop {
        let mut stream = UnixStream::connect(socket)?;
        stream.write_all(b"metrics\n")?;
        stream.flush()?;

        let mut resp = String::new();
        stream.read_to_string(&mut resp)?;

        if is_tty {
            print!("\r{}", resp.trim());
            io::stdout().flush()?;
        } else {
            println!("{}", resp.trim());
        }

        thread::sleep(Duration::from_millis(watch_ms));
    }
}
