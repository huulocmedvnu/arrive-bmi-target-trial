//! NUTS (nuts-rs) for Poisson regression models with analytic gradients.
//! Usage: bayes <draws> <tune> <chains> <out_dir> <model.json>...  All models x chains run in parallel threads.
//! b1: theta = [a, b(p), g]; eta = a + X b + I0 g + off.
//! b2: theta = [a, b(p), slope, log_tau, u(k)]; inter = slope*score + tau*u; eta = a + X b + I inter + off.
//! nb1: as b1 with a negative binomial (NB2) likelihood, theta = [a, b(p), g, log_phi], Var = mu + mu^2/phi.
//! multi: theta = [a, b(p), g(k)], each g_j ~ Normal(g_mu, g_sd); eta = a + X b + I g + off (event study).
use anyhow::Result;
use nuts_rs::{Chain, CpuLogpFunc, CpuMath, CpuMathError, DiagNutsSettings, HasDims, LogpError, Settings};
use nuts_rs::rand::{SeedableRng, rngs::StdRng};
use serde::Deserialize;
use std::{collections::HashMap, fs, io::Write, time::Instant};
use thiserror::Error;
use statrs::function::gamma::{digamma, ln_gamma};

#[derive(Deserialize, Clone)]
struct Data { name: String, mode: String, n: usize, p: usize, k: usize, #[serde(rename = "X")] x: Vec<f64>,
              #[serde(rename = "I")] i: Vec<f64>, y: Vec<f64>, off: Vec<f64>, a_mu: f64, a_sd: f64, b_sd: f64,
              g_mu: f64, g_sd: f64, slope_sd: f64, tau_sd: f64, score: Vec<f64>, init_a: f64 }

#[derive(Debug, Error)]
enum E { #[error("non-finite")] Bad }
impl LogpError for E { fn is_recoverable(&self) -> bool { true } }

#[derive(Clone)]
struct Post { d: Data, eta: Vec<f64> }

impl Post {
    fn dim(&self) -> usize { match self.d.mode.as_str() { "b1" => 2 + self.d.p, "nb1" => 3 + self.d.p, "multi" => 1 + self.d.p + self.d.k, _ => 3 + self.d.p + self.d.k } }
}

impl HasDims for Post {
    fn dim_sizes(&self) -> HashMap<String, u64> {
        HashMap::from([("unconstrained_parameter".to_string(), self.dim() as u64)])
    }
}

impl CpuLogpFunc for Post {
    type LogpError = E;
    type FlowParameters = ();
    type ExpandedVector = Vec<f64>;
    fn dim(&self) -> usize { Post::dim(self) }
    fn logp(&mut self, t: &[f64], g: &mut [f64]) -> Result<f64, E> {
        let d = &self.d; let (n, p, k) = (d.n, d.p, d.k);
        let a = t[0]; let b = &t[1..1 + p];
        let mut inter = vec![0.0; k]; let (mut tau, mut slope, mut u_off) = (0.0, 0.0, 0usize);
        let nb = d.mode == "nb1"; let phi = if nb { t[2 + p].exp() } else { 0.0 }; let mut dphi = 0.0;
        let multi = d.mode == "multi";
        if multi { for j in 0..k { inter[j] = t[1 + p + j]; } } else if d.mode == "b1" || nb { inter[0] = t[1 + p]; } else {
            slope = t[1 + p]; tau = t[2 + p].exp(); u_off = 3 + p;
            for j in 0..k { inter[j] = slope * d.score[j] + tau * t[u_off + j]; }
        }
        let mut ll = 0.0; let mut q = vec![0.0; k];
        for v in g.iter_mut() { *v = 0.0; }
        for r in 0..n {
            let xr = &d.x[r * p..(r + 1) * p]; let ir = &d.i[r * k..(r + 1) * k];
            let mut e = a + d.off[r];
            for j in 0..p { e += xr[j] * b[j]; }
            for j in 0..k { e += ir[j] * inter[j]; }
            self.eta[r] = e;
            let mu = e.exp(); let y = d.y[r];
            let res = if nb {
                ll += ln_gamma(y + phi) - ln_gamma(phi) + phi * (phi / (phi + mu)).ln() + y * (mu / (phi + mu)).ln();
                dphi += digamma(y + phi) - digamma(phi) + (phi / (phi + mu)).ln() + 1.0 - (y + phi) / (phi + mu);
                phi * (y - mu) / (phi + mu)
            } else { ll += y * e - mu; y - mu };
            g[0] += res;
            for j in 0..p { g[1 + j] += res * xr[j]; }
            for j in 0..k { q[j] += res * ir[j]; }
        }
        // priors
        ll += -0.5 * ((a - d.a_mu) / d.a_sd).powi(2); g[0] += -(a - d.a_mu) / (d.a_sd * d.a_sd);
        for j in 0..p { ll += -0.5 * (b[j] / d.b_sd).powi(2); g[1 + j] += -b[j] / (d.b_sd * d.b_sd); }
        if multi {
            for j in 0..k {
                let gg = t[1 + p + j];
                ll += -0.5 * ((gg - d.g_mu) / d.g_sd).powi(2);
                g[1 + p + j] = q[j] - (gg - d.g_mu) / (d.g_sd * d.g_sd);
            }
        } else if d.mode == "b1" || nb {
            let gg = t[1 + p];
            ll += -0.5 * ((gg - d.g_mu) / d.g_sd).powi(2);
            g[1 + p] = q[0] - (gg - d.g_mu) / (d.g_sd * d.g_sd);
            if nb {                                                       // log_phi ~ Normal(3, 3)
                let lp = t[2 + p];
                ll += -0.5 * ((lp - 3.0) / 3.0).powi(2);
                g[2 + p] = dphi * phi - (lp - 3.0) / 9.0;
            }
        } else {
            let mut ds = 0.0; let mut dlt = 0.0;
            for j in 0..k { ds += q[j] * d.score[j]; dlt += q[j] * tau * t[u_off + j]; g[u_off + j] = q[j] * tau - t[u_off + j]; ll += -0.5 * t[u_off + j].powi(2); }
            ll += -0.5 * (slope / d.slope_sd).powi(2); g[1 + p] = ds - slope / (d.slope_sd * d.slope_sd);
            ll += -0.5 * (tau / d.tau_sd).powi(2) + t[2 + p];           // half-normal on tau, log-tau Jacobian
            g[2 + p] = dlt - tau * tau / (d.tau_sd * d.tau_sd) + 1.0;
        }
        if !ll.is_finite() { return Err(E::Bad); }
        Ok(ll)
    }
    fn expand_vector<R: nuts_rs::rand::Rng + ?Sized>(&mut self, _r: &mut R, a: &[f64]) -> Result<Vec<f64>, CpuMathError> { Ok(a.to_vec()) }
}

fn run_chain(d: Data, chain: u64, draws: u64, tune: u64, out: String) -> Result<(String, u64, usize, f64)> {
    let mut s = DiagNutsSettings::default();
    s.num_tune = tune; s.num_draws = draws; s.seed = 20261016 + chain; s.maxdepth = 10;
    let post = Post { eta: vec![0.0; d.n], d: d.clone() };
    let dim = post.dim();
    let mut rng = StdRng::seed_from_u64(20261016 + 1000 * chain);
    let mut c = s.new_chain(chain, CpuMath::new(post), &mut rng)?;
    let mut init = vec![0.0; dim]; init[0] = d.init_a;
    if d.mode == "nb1" { init[dim - 1] = 3.0; }
    c.set_position(&init)?;
    let t0 = Instant::now(); let mut div = 0usize;
    let mut f = fs::File::create(format!("{}/{}_chain{}.csv", out, d.name, chain))?;
    for it in 0..(tune + draws) {
        let (x, info) = c.draw()?;
        if it >= tune {
            if info.diverging { div += 1; }
            let line: Vec<String> = x.iter().map(|v| format!("{:.6}", v)).collect();
            writeln!(f, "{}", line.join(","))?;
        }
    }
    Ok((d.name.clone(), chain, div, t0.elapsed().as_secs_f64()))
}

fn main() -> Result<()> {
    let a: Vec<String> = std::env::args().collect();
    if a[1] == "gradcheck" {                                   // finite-difference check of the analytic gradient
        let d: Data = serde_json::from_str(&fs::read_to_string(&a[2])?)?;
        let mut post = Post { eta: vec![0.0; d.n], d: d.clone() };
        let dim = post.dim(); let mut t = vec![0.05; dim]; t[0] = d.init_a; if d.mode == "nb1" { t[dim - 1] = 2.0; }
        if d.mode == "b2" { t[2 + d.p] = -2.0; }
        let mut g = vec![0.0; dim]; let f0 = CpuLogpFunc::logp(&mut post, &t, &mut g).unwrap();
        let mut worst: f64 = 0.0;
        for j in 0..dim {
            let h = 1e-6; let mut tp = t.clone(); tp[j] += h; let mut tm = t.clone(); tm[j] -= h; let mut gg = vec![0.0; dim];
            let fd = (CpuLogpFunc::logp(&mut post, &tp, &mut gg).unwrap() - CpuLogpFunc::logp(&mut post, &tm, &mut gg).unwrap()) / (2.0 * h);
            let rel = (fd - g[j]).abs() / (1.0 + g[j].abs()); worst = worst.max(rel);
        }
        println!("{} logp {:.3} dim {} max relative gradient error {:.2e}", d.name, f0, dim, worst);
        return Ok(());
    }
    let (draws, tune, chains, out) = (a[1].parse()?, a[2].parse()?, a[3].parse::<u64>()?, a[4].clone());
    let models: Vec<Data> = a[5..].iter().map(|f| serde_json::from_str(&fs::read_to_string(f).unwrap()).unwrap()).collect();
    let t0 = Instant::now();
    let res: Vec<_> = std::thread::scope(|sc| {
        let hs: Vec<_> = models.iter().flat_map(|m| (0..chains).map(move |ch| (m.clone(), ch)))
            .map(|(m, ch)| { let o = out.clone(); sc.spawn(move || run_chain(m, ch, draws, tune, o)) }).collect();
        hs.into_iter().map(|h| h.join().unwrap()).collect()
    });
    for r in res { let (n, c, dv, s) = r?; println!("{}\tchain{}\tdivergences {}\t{:.2}s", n, c, dv, s); }
    println!("total {:.2}s", t0.elapsed().as_secs_f64());
    Ok(())
}
