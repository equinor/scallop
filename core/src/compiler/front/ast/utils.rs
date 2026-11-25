use annotate_snippets::{renderer::DecorStyle, AnnotationKind, Group, Level, Renderer, Snippet};

use super::super::*;
use super::*;

#[derive(Debug, Default, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize)]
pub struct CharLocation {
  pub row: usize,
  pub col: usize,
}

#[derive(Debug, Default, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize)]
pub struct Span<T> {
  pub start: T,
  pub end: T,
}

impl<T> Span<T> {
  pub fn new(start: T, end: T) -> Self {
    Self { start, end }
  }
}

impl Span<usize> {
  pub fn is_default(&self) -> bool {
    self.start == 0 && self.end == 0
  }

  pub fn length(&self) -> usize {
    self.end - self.start
  }
}

#[derive(Clone, Default, PartialEq, Eq, PartialOrd, Ord, Serialize)]
pub struct NodeLocation {
  pub offset_span: Span<usize>,
  pub loc_span: Option<Span<CharLocation>>,
  pub id: Option<usize>,
  pub source_id: usize,
}

impl std::hash::Hash for NodeLocation {
  fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
    state.write_usize(self.source_id);
    state.write_usize(self.offset_span.start);
    state.write_usize(self.offset_span.end);
  }
}

impl NodeLocation {
  /// When cloning a location, we want to keep everything but not the id.
  pub fn clone_without_id(&self) -> Self {
    Self {
      offset_span: self.offset_span.clone(),
      loc_span: self.loc_span.clone(),
      id: None,
      source_id: self.source_id,
    }
  }

  /// Create a location from a single offset span.
  pub fn from_span(start: usize, end: usize) -> Self {
    Self {
      offset_span: Span::new(start, end),
      loc_span: None,
      id: None,
      source_id: 0,
    }
  }

  pub fn error_prefix(&self) -> String {
    match &self.loc_span {
      Some(loc_span) => format!(
        "[{}:{}-{}:{}] ",
        loc_span.start.row, loc_span.start.col, loc_span.end.row, loc_span.end.col
      ),
      None => {
        if self.offset_span.is_default() {
          String::new()
        } else {
          format!("[{}-{}] ", self.offset_span.start, self.offset_span.end)
        }
      }
    }
  }

  pub fn report(&self, src: &Sources) -> String {
    self.render_snippet(src, Level::ERROR)
  }

  pub fn report_warning(&self, src: &Sources) -> String {
    self.render_snippet(src, Level::WARNING)
  }

  fn render_snippet(&self, src: &Sources, level: Level<'_>) -> String {
    let source = &src[self.source_id];
    let content = source.content();
    let start = content.floor_char_boundary(self.offset_span.start.min(content.len()));
    let end = content.ceil_char_boundary(self.offset_span.end.clamp(start, content.len()));
    let mut snippet = Snippet::source(content)
      .line_start(source.first_line_number())
      .annotation(AnnotationKind::Primary.span(start..end));
    if let Some(name) = source.diagnostic_name() {
      snippet = snippet.path(name);
    }
    let renderer = if colored::control::SHOULD_COLORIZE.should_colorize() {
      Renderer::styled()
    } else {
      Renderer::plain()
    };
    renderer
      .decor_style(DecorStyle::Ascii)
      .render(&[Group::with_level(level).element(snippet)])
      .to_string()
  }
}

impl std::fmt::Debug for NodeLocation {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    match (&self.id, &self.loc_span) {
      (None, None) => {
        write!(f, "[{}-{}]", self.offset_span.start, self.offset_span.end)
      }
      (Some(id), None) => {
        write!(f, "[#{}, {}-{}]", id, self.offset_span.start, self.offset_span.end)
      }
      (None, Some(loc_span)) => {
        write!(
          f,
          "[{}:{}-{}:{}]",
          loc_span.start.row, loc_span.start.col, loc_span.end.row, loc_span.end.col,
        )
      }
      (Some(id), Some(loc_span)) => {
        write!(
          f,
          "[#{}, {}:{}-{}:{}]",
          id, loc_span.start.row, loc_span.start.col, loc_span.end.row, loc_span.end.col,
        )
      }
    }
  }
}

/// An AST Node trait
pub trait AstNode: Clone {
  /// Obtain a location of the AstNode
  fn location(&self) -> &NodeLocation;

  /// Obtain a mutable location of the AstNode
  fn location_mut(&mut self) -> &mut NodeLocation;

  /// Clone with a given location
  fn clone_with_loc(&self, loc: NodeLocation) -> Self;
}

#[derive(Clone, Debug, PartialEq, Hash, Serialize)]
pub struct AstNodeWrapper<T> {
  pub _loc: NodeLocation,
  pub _node: T,
}

impl<T: Clone> AstNodeWrapper<T> {
  pub fn clone_without_location_id(&self) -> Self {
    Self {
      _loc: self._loc.clone_without_id(),
      _node: self._node.clone(),
    }
  }

  pub fn location_id(&self) -> Option<usize> {
    self._loc.id
  }

  pub fn location_source_id(&self) -> usize {
    self._loc.source_id
  }
}

impl<T: Clone> AstNode for AstNodeWrapper<T> {
  fn location(&self) -> &NodeLocation {
    &self._loc
  }

  fn location_mut(&mut self) -> &mut NodeLocation {
    &mut self._loc
  }

  fn clone_with_loc(&self, loc: NodeLocation) -> Self {
    Self {
      _loc: loc,
      _node: self._node.clone(),
    }
  }
}

#[allow(unused)]
pub trait NodeVisitor<N> {
  fn visit(&mut self, node: &N) {}

  fn visit_mut(&mut self, node: &mut N) {}
}

#[allow(unused)]
impl<U, V> NodeVisitor<V> for U {
  default fn visit(&mut self, node: &V) {}

  default fn visit_mut(&mut self, node: &mut V) {}
}

impl<T, V> NodeVisitor<V> for Vec<T> {
  fn visit(&mut self, node: &V) {
    for elem in self {
      elem.visit(node);
    }
  }

  fn visit_mut(&mut self, node: &mut V) {
    for elem in self {
      elem.visit_mut(node);
    }
  }
}

impl<T, V> NodeVisitor<V> for Box<T> {
  fn visit(&mut self, node: &V) {
    (&**self).visit(node)
  }

  fn visit_mut(&mut self, node: &mut V) {
    (**self).visit_mut(node)
  }
}

macro_rules! impl_node_visitor_tuple {
  ( $($id:ident : $index:tt),+ ) => {
    impl<Node, $($id,)*> NodeVisitor<Node> for ($(&mut $id,)*) {
      default fn visit(&mut self, node: &Node) {
        $( <$id as NodeVisitor<Node>>::visit(self.$index, node); )*
      }
      default fn visit_mut(&mut self, node: &mut Node) {
        $( <$id as NodeVisitor<Node>>::visit_mut(self.$index, node); )*
      }
    }
  };
}

macro_rules! impl_node_visitor_tuples {
  ([$($done:ident : $done_index:tt,)*] $head:ident : $index:tt $(, $tail:ident : $tail_index:tt)*) => {
    impl_node_visitor_tuple!($($done : $done_index,)* $head : $index);
    impl_node_visitor_tuples!([$($done : $done_index,)* $head : $index,] $($tail : $tail_index),*);
  };
  ([$($done:tt)*]) => {};
}

impl_node_visitor_tuples!(
  [] A: 0, B: 1, C: 2, D: 3, E: 4, F: 5, G: 6, H: 7, I: 8, J: 9, K: 10,
  L: 11, M: 12, N: 13, O: 14, P: 15, Q: 16, R: 17, S: 18, T: 19, U: 20, V: 21
);

#[allow(unused)]
pub trait AstWalker {
  fn walk<V>(&self, v: &mut V);

  fn walk_mut<V>(&mut self, v: &mut V);
}

macro_rules! derive_ast_walker {
  ($ty:ty) => {
    impl AstWalker for $ty {
      fn walk<V>(&self, _: &mut V) {}

      fn walk_mut<V>(&mut self, _: &mut V) {}
    }
  };
}

derive_ast_walker!(i8);
derive_ast_walker!(i16);
derive_ast_walker!(i32);
derive_ast_walker!(i64);
derive_ast_walker!(i128);
derive_ast_walker!(isize);
derive_ast_walker!(u8);
derive_ast_walker!(u16);
derive_ast_walker!(u32);
derive_ast_walker!(u64);
derive_ast_walker!(u128);
derive_ast_walker!(usize);
derive_ast_walker!(f32);
derive_ast_walker!(f64);
derive_ast_walker!(bool);
derive_ast_walker!(char);
derive_ast_walker!(String);
derive_ast_walker!(crate::common::input_tag::DynamicInputTag);
derive_ast_walker!(crate::common::binary_op::BinaryOp);

impl<T> AstWalker for Vec<T>
where
  T: AstWalker,
{
  fn walk<V>(&self, v: &mut V) {
    for child in self {
      child.walk(v)
    }
  }

  fn walk_mut<V>(&mut self, v: &mut V) {
    for child in self {
      child.walk_mut(v)
    }
  }
}

impl<T> AstWalker for Option<T>
where
  T: AstWalker,
{
  fn walk<V>(&self, v: &mut V) {
    if let Some(n) = self {
      n.walk(v)
    }
  }

  fn walk_mut<V>(&mut self, v: &mut V) {
    if let Some(n) = self {
      n.walk_mut(v)
    }
  }
}

impl<T> AstWalker for Box<T>
where
  T: AstWalker,
{
  fn walk<V>(&self, v: &mut V) {
    (**self).walk(v)
  }

  fn walk_mut<V>(&mut self, v: &mut V) {
    (**self).walk_mut(v)
  }
}

impl<A> AstWalker for (A,)
where
  A: AstWalker,
{
  fn walk<V>(&self, v: &mut V) {
    self.0.walk(v);
  }

  fn walk_mut<V>(&mut self, v: &mut V) {
    self.0.walk_mut(v);
  }
}

impl<A, B> AstWalker for (A, B)
where
  A: AstWalker,
  B: AstWalker,
{
  fn walk<V>(&self, v: &mut V) {
    self.0.walk(v);
    self.1.walk(v);
  }

  fn walk_mut<V>(&mut self, v: &mut V) {
    self.0.walk_mut(v);
    self.1.walk_mut(v);
  }
}

impl<A, B, C> AstWalker for (A, B, C)
where
  A: AstWalker,
  B: AstWalker,
  C: AstWalker,
{
  fn walk<V>(&self, v: &mut V) {
    self.0.walk(v);
    self.1.walk(v);
    self.2.walk(v);
  }

  fn walk_mut<V>(&mut self, v: &mut V) {
    self.0.walk_mut(v);
    self.1.walk_mut(v);
    self.2.walk_mut(v);
  }
}
