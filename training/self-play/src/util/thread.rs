use crossbeam::channel::{Receiver, Sender};
use crossbeam::utils::Backoff;
use std::any::Any;
use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

struct ManagerInner {
    termination_senders: HashMap<u64, Sender<()>>,
    termination_sender_next_id: u64,
}

pub(crate) struct ThreadManager {
    inner: Arc<Mutex<ManagerInner>>,
    threads: Vec<Thread>,
}

impl ThreadManager {
    pub fn new() -> Self {
        Self {
            inner: Arc::new(Mutex::new(ManagerInner {
                termination_senders: Default::default(),
                termination_sender_next_id: 0,
            })),
            threads: Vec::new(),
        }
    }

    pub fn spawn_thread(&mut self, name: impl AsRef<str>, thread_main: impl FnOnce(ThreadControl) + Send + 'static) {
        let name = name.as_ref().to_string();
        let ready_flag = ReadyFlag::new();
        let threads_control = ThreadControl::new(ready_flag.clone(), self.inner.clone());
        let (join_sender, join_receiver) = crossbeam::channel::bounded(1);

        let handle = thread::Builder::new()
            .name(name.to_string())
            .spawn({
                let name = name.clone();
                move || {
                    thread_main(threads_control);

                    let send_err = join_sender.try_send(());
                    if let Err(e) = send_err {
                        log::warn!("Failed to send join signal from thread '{}': {}", name, e);
                    }
                }
            })
            .unwrap();

        let thread = Thread {
            name,
            handle,
            ready_flag,
            join_receiver,
        };

        self.threads.push(thread);
    }

    pub(crate) fn wait_ready(&self, deadline: Instant) {
        for thread in self.threads.iter() {
            thread
                .ready_flag
                .wait_ready(deadline)
                .unwrap_or_else(|e| panic!("Thread '{}': {}", thread.name, e));
        }
    }

    pub fn terminate(self) -> Result<(), Box<dyn Any + Send + 'static>> {
        let mut join_err = None; // return the first error, if any

        {
            let senders = &mut self.inner.lock().unwrap();
            for sender in senders.termination_senders.values_mut() {
                if let Err(e) = sender.try_send(()) {
                    log::warn!("Failed to send stop signal to thread: {}", e);
                }
            }

            // dont hold he lock while joining threads
        }

        for handler in self.threads.into_iter().rev() {
            log::debug!("Joining thread '{}'", handler.name());
            let join_res = handler.join();
            join_err = join_err.or(join_res.err());
        }
        join_err.map(Err).unwrap_or(Ok(()))
    }

    pub fn any_thread_crashed(&mut self) -> bool {
        let mut any_crash = false;
        self.threads.retain(|handler: &Thread| {
            if !handler.handle.is_finished() {
                true // Thread is still running, keep it
            } else if handler.join_receiver.try_recv().is_ok() {
                log::debug!("Thread '{}' has terminated", handler.name());
                false // Thread has (successfully) terminated, remove
            } else {
                log::error!("Thread '{}' has crashed", handler.name());
                any_crash = true;
                true // Thread has crashed, keep it in the list
            }
        });
        any_crash
    }
}

struct Thread {
    name: String,
    handle: JoinHandle<()>,
    ready_flag: ReadyFlag,
    join_receiver: Receiver<()>,
}

impl Thread {
    fn name(&self) -> &str {
        &self.name
    }

    fn join(self) -> Result<(), Box<dyn Any + Send + 'static>> {
        let join_res = match self.join_receiver.recv_timeout(Duration::from_secs(7)) {
            Ok(()) => self.handle.join(),
            Err(crossbeam::channel::RecvTimeoutError::Disconnected) => {
                let t0 = Instant::now();
                loop {
                    thread::sleep(Duration::from_millis(10));
                    if self.handle.is_finished() {
                        break self.handle.join();
                    }
                    if t0.elapsed() < Duration::from_millis(100) {
                        panic!("Thread '{}' was disconnected but still running", self.name);
                    }
                }
            }
            Err(crossbeam::channel::RecvTimeoutError::Timeout) => {
                let msg = format!("Thread '{}' join timeout after 7 secs", self.name);
                log::error!("{msg}");
                return Err(Box::new(msg));
            }
        };
        if join_res.is_err() {
            log::error!("Thread '{}' panicked", self.name);
        }
        join_res
    }
}

pub struct ThreadControl {
    id: u64,
    manager: Arc<Mutex<ManagerInner>>,

    // communication from the thread to the manager
    ready_flag: ReadyFlag,
    // communication from the manager to the thread
    termination_receiver: Receiver<()>,
}

impl ThreadControl {
    fn new(ready_flag: ReadyFlag, manager: Arc<Mutex<ManagerInner>>) -> Self {
        let (sender, termination_receiver) = crossbeam::channel::bounded(1);
        let id = {
            let manager = &mut manager.lock().unwrap();

            let next_id = &mut manager.termination_sender_next_id;
            let id = *next_id;
            *next_id += 1;

            let senders_map = &mut manager.termination_senders;
            assert!(!senders_map.contains_key(&id));
            senders_map.insert(id, sender);
            assert!(senders_map.len() < 3000, "ThreadsControl cloned too many times.");

            id
        };

        Self {
            id,
            manager,
            ready_flag,
            termination_receiver,
        }
    }

    pub fn set_ready(&self) {
        self.ready_flag.set_ready();
    }

    pub fn termination_receiver(&self) -> &Receiver<()> {
        &self.termination_receiver
    }
}

impl Clone for ThreadControl {
    fn clone(&self) -> Self {
        ThreadControl::new(self.ready_flag.clone(), self.manager.clone())
    }
}

impl Drop for ThreadControl {
    fn drop(&mut self) {
        let senders = &mut self.manager.lock().unwrap().termination_senders;
        senders.remove(&self.id).unwrap();
    }
}

#[derive(Clone)]
struct ReadyFlag(Arc<ReadyFlagInner>);
struct ReadyFlagInner {
    flag: AtomicBool,
}

impl Default for ReadyFlag {
    fn default() -> Self {
        Self::new()
    }
}

impl ReadyFlag {
    pub fn new() -> Self {
        Self(Arc::new(ReadyFlagInner {
            flag: AtomicBool::new(false),
        }))
    }

    pub fn set_ready(&self) {
        self.0
            .flag
            .compare_exchange(false, true, Ordering::Release, Ordering::Relaxed)
            .expect("Ready flag already set");
    }

    pub fn wait_ready(&self, deadline: Instant) -> Result<(), String> {
        let backoff = Backoff::new();
        while !self.0.flag.load(Ordering::Acquire) {
            if Instant::now() > deadline {
                return Err(String::from("Timeout waiting for ready flag"));
            }
            if !backoff.is_completed() {
                backoff.snooze();
            } else {
                thread::sleep(Duration::from_millis(50));
            }
        }
        Ok(())
    }
}
