use std::fs;
use std::io;
use std::path::{Path, PathBuf};

pub struct PidFile {
    path: PathBuf,
}

impl PidFile {
    pub fn create(path: &Path) -> io::Result<Self> {
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)?;
        }

        if path.exists() {
            if let Ok(contents) = fs::read_to_string(path) {
                if let Ok(pid) = contents.trim().parse::<i32>() {
                    if unsafe { libc::kill(pid, 0) } == 0 {
                        return Err(io::Error::new(
                            io::ErrorKind::AlreadyExists,
                            format!("Daemon already running (PID: {})", pid),
                        ));
                    }
                }
            }
            fs::remove_file(path)?;
        }

        let pid = unsafe { libc::getpid() };
        fs::write(path, format!("{}\n", pid))?;

        Ok(Self {
            path: path.to_path_buf(),
        })
    }
}

impl Drop for PidFile {
    fn drop(&mut self) {
        let _ = fs::remove_file(&self.path);
    }
}
