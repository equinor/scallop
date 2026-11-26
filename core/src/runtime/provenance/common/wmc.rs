use super::{BooleanFormula, Semiring};

#[cfg(not(target_family = "wasm"))]
mod oxidd_backend;
#[cfg(target_family = "wasm")]
mod rsdd_backend;

#[cfg(not(target_family = "wasm"))]
use oxidd_backend as default_backend;

/// A probability model over the fact IDs used by an event.
pub trait WmcModel<S: Semiring> {
  fn count(&self, event: &BooleanFormula, algebra: &S, backend: &impl WmcBackend) -> S::Element;
}

/// Count an event under a probability model, using the given numeric algebra.
pub fn count<S: Semiring>(
  event: &BooleanFormula,
  model: &impl WmcModel<S>,
  algebra: &S,
  backend: &impl WmcBackend,
) -> S::Element {
  model.count(event, algebra, backend)
}

/// Independent Bernoulli facts, indexed by their original fact IDs.
pub struct IndependentFacts<W>(pub W);

impl<S, W> WmcModel<S> for IndependentFacts<W>
where
  S: Semiring,
  W: Fn(&usize) -> S::Element,
{
  fn count(&self, event: &BooleanFormula, algebra: &S, backend: &impl WmcBackend) -> S::Element {
    backend.wmc(event, algebra, &self.0)
  }
}
#[cfg(target_family = "wasm")]
use rsdd_backend as default_backend;

/// Adapter for a weighted model counting engine. No decision-diagram types are shared with Scallop.
///
/// Pass an implementation to `new_with_wmc_backend` on a proof-based provenance, or use
/// [`super::AsBooleanFormula::wmc_using`] to count a formula with it directly.
pub trait WmcBackend: Clone + 'static {
  /// Count satisfying assignments using normalized literal weights.
  ///
  /// `weight` uses the original (possibly sparse) fact IDs and returns the positive weight.
  /// Obtain negative weights with [`Semiring::negate`]. A variable omitted by a reduced
  /// diagram contributes one, since its two weights sum to one. Values may own gradients
  /// and need not implement `Copy`.
  fn wmc<S: Semiring>(
    &self,
    formula: &BooleanFormula,
    semiring: &S,
    weight: &impl Fn(&usize) -> S::Element,
  ) -> S::Element;
}

/// Scallop's default backend: OxiDD on native targets, RSDD on WebAssembly where OxiDD
/// cannot spawn its manager threads.
#[derive(Clone, Copy, Debug, Default)]
pub struct DefaultWmcBackend;

impl WmcBackend for DefaultWmcBackend {
  fn wmc<S: Semiring>(
    &self,
    formula: &BooleanFormula,
    semiring: &S,
    weight: &impl Fn(&usize) -> S::Element,
  ) -> S::Element {
    default_backend::wmc(formula, semiring, weight)
  }
}
