use std::sync::atomic::{AtomicU8, Ordering};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::{Duration, Instant};

#[derive(Copy, Clone, Eq, PartialEq, Debug)]
#[repr(u8)]
pub(crate) enum BatchState {
    /* The batch is currently collecting samples, until it has batch_size of them */
    Collect,
    /* The batch is currently computed by one of the threads */
    Compute,
    /* The batch computation is done, the results are available to all threads */
    Done,
}
struct Batch<I, O> {
    inner: Mutex<BatchInner<I, O>>,
    state: AtomicU8,
}
enum BatchInner<I, O> {
    Collect(Vec<I>),
    Compute,
    Done(Vec<Option<O>>),
}
impl<I, O> Batch<I, O> {
    fn new() -> Self {
        Self {
            inner: Mutex::new(BatchInner::Collect(Vec::new())),
            state: AtomicU8::new(BatchState::Collect as u8),
        }
    }
}

pub(crate) struct Batcher<I, O> {
    next_batch: Mutex<Arc<Batch<I, O>>>,
    batch_size: usize,
}

impl<I, O> Batcher<I, O> {
    pub fn new(batch_size: usize) -> Self {
        Self {
            next_batch: Mutex::new(Arc::new(Batch::new())),
            batch_size,
        }
    }

    pub fn apply(&self, input: I, deadline: Duration, apply_impl: impl FnOnce(Vec<I>) -> Vec<O>) -> O {
        if self.batch_size <= 1 {
            let outputs = apply_impl(vec![input]);
            let [output] = outputs.try_into().unwrap_or_else(|_| unreachable!());
            return output;
        }
        let apply_impl = |xs| apply_impl(xs).into_iter().map(Some).collect::<Vec<_>>();

        let backoff = crossbeam_utils::Backoff::new();
        let (batch_ptr, input_idx) = loop {
            {
                let mut next_batch = self.next_batch.lock().unwrap();
                let batch_ptr = next_batch.clone();
                let mut batch = batch_ptr.inner.lock().unwrap();

                if let BatchInner::Collect(batch_samples) = &mut *batch {
                    let input_idx = batch_samples.len();
                    batch_samples.push(input);
                    if batch_samples.len() < self.batch_size {
                        drop(batch);
                        break (batch_ptr, input_idx);
                    }

                    batch_ptr
                        .state
                        .compare_exchange(
                            BatchState::Collect as u8,
                            BatchState::Compute as u8,
                            Ordering::SeqCst,
                            Ordering::SeqCst,
                        )
                        .unwrap(); // can not fail, only who has the batch inner lock can change the state to 'compute'
                    *next_batch = Arc::new(Batch::new());
                    drop(next_batch);

                    let inputs = std::mem::replace(&mut *batch, BatchInner::Compute);
                    let BatchInner::Collect(inputs) = inputs else {
                        unreachable!()
                    };
                    drop(batch);

                    let mut outputs = apply_impl(inputs);
                    let output = outputs[input_idx].take().unwrap();
                    {
                        let mut batch = batch_ptr.inner.lock().unwrap();
                        *batch = BatchInner::Done(outputs);
                    };
                    batch_ptr
                        .state
                        .compare_exchange(
                            BatchState::Compute as u8,
                            BatchState::Done as u8,
                            Ordering::SeqCst,
                            Ordering::SeqCst,
                        )
                        .unwrap(); // can not fail, only the computed thread set 'done' state
                    return output;
                }
            }

            if !backoff.is_completed() {
                backoff.snooze();
            } else {
                thread::sleep(Duration::from_micros(100));
            }
        };

        let wait_start_time = Instant::now();
        loop {
            thread::sleep(Duration::from_micros(500));

            match batch_ptr.state.load(Ordering::SeqCst) {
                s if s == BatchState::Done as u8 => {
                    let mut batch = batch_ptr.inner.lock().unwrap();
                    let BatchInner::Done(outputs) = &mut *batch else {
                        unreachable!()
                    };
                    return outputs[input_idx].take().unwrap();
                }
                s if s == BatchState::Collect as u8 => {
                    if wait_start_time.elapsed() < deadline {
                        continue;
                    }

                    let mut next_batch = self.next_batch.lock().unwrap();

                    let mut batch = batch_ptr.inner.lock().unwrap();
                    if !matches!(&*batch, BatchInner::Collect(_)) {
                        continue;
                    }
                    batch_ptr
                        .state
                        .compare_exchange(
                            BatchState::Collect as u8,
                            BatchState::Compute as u8,
                            Ordering::SeqCst,
                            Ordering::SeqCst,
                        )
                        .unwrap(); // can not fail, only who has the batch inner lock can change the state to 'compute'
                    {
                        if Arc::ptr_eq(&*next_batch, &batch_ptr) {
                            *next_batch = Arc::new(Batch::new());
                        }
                    }

                    let inputs = std::mem::replace(&mut *batch, BatchInner::Compute);
                    let BatchInner::Collect(inputs) = inputs else {
                        unreachable!()
                    };
                    drop(batch);

                    let mut outputs = apply_impl(inputs);
                    let output = outputs[input_idx].take().unwrap();
                    {
                        let mut batch = batch_ptr.inner.lock().unwrap();
                        *batch = BatchInner::Done(outputs);
                    };
                    batch_ptr
                        .state
                        .compare_exchange(
                            BatchState::Compute as u8,
                            BatchState::Done as u8,
                            Ordering::SeqCst,
                            Ordering::SeqCst,
                        )
                        .unwrap(); // can not fail, only the computed thread set 'done' state
                    return output;
                }
                s if s == BatchState::Compute as u8 => {}
                _ => unreachable!(),
            }
        }
    }

    pub fn batch_size(&self) -> usize {
        self.batch_size
    }
}
