pub(crate) struct RunningAverage {
    value: f64,
    epsilon: f64,
    inner: metrics::Gauge,
}
impl RunningAverage {
    pub fn new(epsilon: f64, inner: metrics::Gauge) -> Self {
        assert!((0.0..1.0).contains(&epsilon));
        Self {
            value: 0.0,
            epsilon,
            inner,
        }
    }

    pub fn set(&mut self, new_value: f64) {
        self.value = (1.0 - self.epsilon) * self.value + self.epsilon * new_value;
        self.inner.set(self.value);
    }

    pub fn get(&self) -> f64 {
        self.value
    }
}
