//! AstNode can be annotated on three kinds of rust constructs
//!
//! 1. A struct
//!
//! In this case, the structs should be named with an underscore `_` in front
//!
//! ``` ignore
//! #[derive(AstNode)]
//! struct _MyNode {
//!   field1: T1,
//!   field2: Box<T2>,
//!   field3: Vec<T3>,
//!   field4: Option<T4>,
//! }
//! ```
//!
//! The struct `_MyNode` should be annotated with `AstNode`.
//! When used, one should use the regular type without the underscore `MyNode`.
//! Each field can have types wrapped by `Box`, `Vec`, and `Option`, in which case constructors, getters, and setters are handled differently.
//! There is no restrictions on the field types.
//!
//! You can create `MyNode` with a few constructors
//!
//! ``` ignore
//! MyNode::new(field1: T1, field2: T2, field3: Vec<T3>, field4: Option<T4>)
//! MyNode::new_with_loc(field1: T1, field2: T2, field3: Vec<T3>, field4: Option<T4>, loc: NodeLocation)
//! MyNode::new_with_span(field1: T1, field2: T2, field3: Vec<T3>, field4: Option<T4>, start: usize, end: usize)
//! ```
//!
//! Please notice that for `field2` you don't need to wrap the type inside of `Box`.
//! Use `#[ast_node_no_span_constructor]` to omit `new_with_span` on a struct. The
//! inner node's `with_span` and the wrapper's `with_node_and_span` remain available.
//!
//! You can access each field using functions like `field1() -> &T1` or `field1_mut() -> &mut T1`.
//! For `Box` fields, the accessors will skip the `Box`: `field2() -> &T2`, `field2_mut() -> &mut T2`.
//! For `Vec` fields, there are functions like `iter_field3() -> Iterator`, `iter_field3_mut() -> Iterator`, `num_field3() -> usize`, and `has_field3() -> bool`.
//! For `Option` fields, there is a function `has_field4() -> bool`.
//!
//! 2. A variant enum
//!
//! In this case, `AstNode` should annotate an enum that is named normally, with all variants having just a single argument
//!
//! ``` ignore
//! #[derive(AstNode)]
//! enum EnumNode {
//!   V1(T1),
//!   V2(T2),
//! }
//! ```
//!
//! We assume that all types inside of the variants are (annotated by) `AstNode`.
//! As such, we have provided the following helper functions
//!
//! ``` ignore
//! EnumNode::is_v1(&self) -> bool // there will be an `is_*` function for each variant
//! EnumNode::as_v1(&self) -> Option<&V1> // there will be an `as_*` function for each variant
//! ```
//!
//! 3. A terminal enum
//!
//! In this case, `AstNode` should annotate an enum that is named also with an underscore in front.
//! There should be at least one variant having no argument; all variants should not have more than one arguments.
//!
//! ``` ignore
//! #[derive(AstNode)]
//! enum _TerminalNode {
//!   S1,
//!   S2,
//!   S3(T3),
//! }
//! ```
//!
//! All operations should be done on the normal type `TerminalNode`.
//! We offer the following helper functions
//!
//! ``` ignore
//! TerminalNode::s1() -> Self // constructor without location
//! TerminalNode::s1_with_loc(loc: NodeLocation) // constructor with location provided
//! TerminalNode::s1_with_span(start: usize, end: usize) // constructor with location span provided
//! TerminalNode::s3(t3: T3) -> Self // providing the argument to construct variant s3
//! TerminalNode::s3_with_loc(t3: T3, loc: NodeLocation) // providing the argument to construct variant s3 with location provided
//! TerminalNode::s3_with_span(t3: T3, start: usize, end: usize) // providing the argument to construct variant s3 with location span provided
//! TerminalNode::is_s1(&self) -> bool // there is an `is_*` function for each variant
//! TerminalNode::as_s3(&self) -> Option<&T3> // there is an `as_*` function if that variant has an argument
//! ```

use proc_macro::TokenStream;
use proc_macro2::{Ident, TokenStream as TokenStream2};
use quote::{format_ident, quote};
use syn::{parse_macro_input, Data, DeriveInput, Fields, GenericArgument, Meta, PathArguments, Type};

/// Derive a Scallop AST node, including its location and traversal methods.
#[proc_macro_derive(AstNode, attributes(ast_node_no_span_constructor))]
pub fn derive_ast_node(tokens: TokenStream) -> TokenStream {
  let input = parse_macro_input!(tokens as DeriveInput);
  expand(&input).unwrap_or_else(syn::Error::into_compile_error).into()
}

fn expand(input: &DeriveInput) -> syn::Result<TokenStream2> {
  let inner = &input.ident;
  if !input.generics.params.is_empty() || input.generics.where_clause.is_some() {
    return Err(syn::Error::new(inner.span(), "AstNode does not support generic nodes"));
  }
  let omit_span_constructor = input
    .attrs
    .iter()
    .find(|attr| matches!(&attr.meta, Meta::Path(path) if path.is_ident("ast_node_no_span_constructor")));

  match &input.data {
    Data::Struct(data) => {
      if matches!(&data.fields, Fields::Unnamed(_)) {
        return Err(syn::Error::new(inner.span(), "AstNode structs require named fields"));
      }
      let outer = wrapper_name(inner)?;
      Ok(derive_struct(
        input,
        &outer,
        &data.fields,
        omit_span_constructor.is_some(),
      ))
    }
    Data::Enum(data) => {
      if let Some(attr) = omit_span_constructor {
        return Err(syn::Error::new_spanned(
          attr,
          "ast_node_no_span_constructor is only valid on structs",
        ));
      }
      if data.variants.is_empty() {
        return Err(syn::Error::new(
          inner.span(),
          "AstNode enums require at least one variant",
        ));
      }
      let variants = data
        .variants
        .iter()
        .map(|variant| {
          let ty = match &variant.fields {
            Fields::Unit if variant.discriminant.is_none() => None,
            Fields::Unnamed(fields) if fields.unnamed.len() == 1 => Some(&fields.unnamed[0].ty),
            _ => {
              return Err(syn::Error::new(
                variant.ident.span(),
                "AstNode variants must be unit variants without a discriminant or have one unnamed field",
              ))
            }
          };
          Ok((&variant.ident, ty))
        })
        .collect::<syn::Result<Vec<_>>>()?;

      if variants.iter().any(|(_, ty)| ty.is_none()) {
        let outer = wrapper_name(inner)?;
        let wrapper = wrapped_node(input, &outer, &[]);
        let helpers = variants
          .into_iter()
          .map(|(variant, ty)| terminal_methods(inner, variant, ty));
        Ok(quote! {
          #wrapper
          impl #outer { #(#helpers)* }
        })
      } else {
        Ok(derive_variant_enum(
          inner,
          variants.into_iter().map(|(name, ty)| (name, ty.unwrap())),
        ))
      }
    }
    Data::Union(_) => Err(syn::Error::new(
      inner.span(),
      "AstNode can only be derived for a struct or enum",
    )),
  }
}

fn wrapper_name(inner: &Ident) -> syn::Result<Ident> {
  let name = inner.to_string();
  let outer = name
    .strip_prefix('_')
    .ok_or_else(|| syn::Error::new(inner.span(), "Wrapped AstNode names must begin with an underscore"))?;
  syn::parse_str::<Ident>(outer)
    .map(|mut ident| {
      ident.set_span(inner.span());
      ident
    })
    .map_err(|error| syn::Error::new(inner.span(), error))
}

fn wrapped_node(input: &DeriveInput, outer: &Ident, fields: &[&Ident]) -> TokenStream2 {
  let inner = &input.ident;
  let vis = &input.vis;
  quote! {
    #vis type #outer = AstNodeWrapper<#inner>;

    impl #inner {
      pub fn with_location(self, loc: NodeLocation) -> #outer {
        #outer { _loc: loc, _node: self }
      }
      pub fn with_span(self, start: usize, end: usize) -> #outer {
        self.with_location(NodeLocation::from_span(start, end))
      }
    }

    impl #outer {
      pub fn internal(&self) -> &#inner { &self._node }
      pub fn internal_mut(&mut self) -> &mut #inner { &mut self._node }
    }

    impl AstWalker for #outer {
      fn walk<V>(&self, v: &mut V) {
        v.visit(self);
        v.visit(&self._loc);
        #(self._node.#fields.walk(v);)*
      }
      fn walk_mut<V>(&mut self, v: &mut V) {
        v.visit_mut(self);
        v.visit_mut(&mut self._loc);
        #(self._node.#fields.walk_mut(v);)*
      }
    }
  }
}

fn derive_struct(input: &DeriveInput, outer: &Ident, fields: &Fields, omit_span_constructor: bool) -> TokenStream2 {
  let inner = &input.ident;
  let names: Vec<_> = fields.iter().map(|field| field.ident.as_ref().unwrap()).collect();
  let wrapper = wrapped_node(input, outer, &names);
  let mut args = Vec::new();
  let mut init = Vec::new();
  let mut accessors = Vec::new();
  for (field, name) in fields.iter().zip(names) {
    let vis = &field.vis;
    let ty = &field.ty;
    let name_mut = format_ident!("{name}_mut");
    let arg_type = if let Some(ty) = type_arg(ty, "Box") {
      init.push(quote!(#name: Box::new(#name)));
      ty
    } else {
      init.push(quote!(#name));
      ty
    };
    args.push(quote!(#name: #arg_type));
    accessors.push(quote! {
      #vis fn #name(&self) -> &#arg_type { &self._node.#name }
      #vis fn #name_mut(&mut self) -> &mut #arg_type { &mut self._node.#name }
    });

    let has = format_ident!("has_{name}");
    if let Some(elem) = type_arg(ty, "Vec") {
      let num = format_ident!("num_{name}");
      let iter = format_ident!("iter_{name}");
      let iter_mut = format_ident!("iter_{name}_mut");
      accessors.push(quote! {
        #vis fn #has(&self) -> bool { !self.#name().is_empty() }
        #vis fn #num(&self) -> usize { self.#name().len() }
        #vis fn #iter(&self) -> impl Iterator<Item = &#elem> { self.#name().iter() }
        #vis fn #iter_mut(&mut self) -> impl Iterator<Item = &mut #elem> { self.#name_mut().iter_mut() }
      });
    } else if type_arg(ty, "Option").is_some() {
      accessors.push(quote!(#vis fn #has(&self) -> bool { self.#name().is_some() }));
    }
  }

  let with_span = (!omit_span_constructor).then(|| {
    quote! {
      pub fn new_with_span(#(#args,)* start: usize, end: usize) -> Self {
        Self { _loc: NodeLocation::from_span(start, end), _node: #inner { #(#init,)* } }
      }
    }
  });
  quote! {
    #wrapper
    impl #inner {
      pub fn new(#(#args),*) -> Self { Self { #(#init,)* } }
    }
    impl #outer {
      pub fn new(#(#args),*) -> Self {
        Self { _loc: NodeLocation::default(), _node: #inner { #(#init,)* } }
      }
      pub fn new_with_loc(#(#args,)* _loc: NodeLocation) -> Self {
        Self { _loc, _node: #inner { #(#init,)* } }
      }
      #with_span
      pub fn new_with_node_and_loc(node: #inner, loc: NodeLocation) -> Self {
        Self { _loc: loc, _node: node }
      }
      pub fn with_node_and_span(node: #inner, start: usize, end: usize) -> Self {
        Self::new_with_node_and_loc(node, NodeLocation::from_span(start, end))
      }
      #(#accessors)*
    }
  }
}

fn type_arg<'a>(ty: &'a Type, wrapper: &str) -> Option<&'a Type> {
  if let Type::Path(ty) = ty
    && ty.qself.is_none()
    && ty.path.leading_colon.is_none()
    && ty.path.segments.len() == 1
    && ty.path.segments[0].ident == wrapper
    && let PathArguments::AngleBracketed(args) = &ty.path.segments[0].arguments
    && args.args.len() == 1
    && let GenericArgument::Type(arg) = &args.args[0]
  {
    Some(arg)
  } else {
    None
  }
}

fn derive_variant_enum<'a>(name: &Ident, variants: impl Iterator<Item = (&'a Ident, &'a Type)>) -> TokenStream2 {
  let (variants, types): (Vec<_>, Vec<_>) = variants.unzip();
  let methods = variants.iter().zip(types).map(|(variant, ty)| {
    let ctor = format_ident!("{}", variant.to_string().to_lowercase(), span = variant.span());
    let is = format_ident!("is_{ctor}");
    let as_variant = format_ident!("as_{ctor}");
    quote! {
      pub fn #ctor(v: #ty) -> Self { Self::#variant(v) }
      pub fn #is(&self) -> bool { matches!(self, Self::#variant(_)) }
      pub fn #as_variant(&self) -> Option<&#ty> {
        match self { Self::#variant(v) => Some(v), _ => None }
      }
    }
  });
  quote! {
    impl #name { #(#methods)* }

    impl AstNode for #name {
      fn location(&self) -> &NodeLocation {
        match self { #(Self::#variants(v) => v.location(),)* }
      }
      fn location_mut(&mut self) -> &mut NodeLocation {
        match self { #(Self::#variants(v) => v.location_mut(),)* }
      }
      fn clone_with_loc(&self, loc: NodeLocation) -> Self {
        match self { #(Self::#variants(v) => Self::#variants(v.clone_with_loc(loc)),)* }
      }
    }

    impl AstWalker for #name {
      fn walk<V>(&self, v: &mut V) {
        v.visit(self);
        match self { #(Self::#variants(c) => c.walk(v),)* }
      }
      fn walk_mut<V>(&mut self, v: &mut V) {
        v.visit_mut(self);
        match self { #(Self::#variants(c) => c.walk_mut(v),)* }
      }
    }
  }
}

fn terminal_methods(inner: &Ident, variant: &Ident, ty: Option<&Type>) -> TokenStream2 {
  let ctor = format_ident!("{}", variant.to_string().to_lowercase(), span = variant.span());
  let with_loc = format_ident!("{ctor}_with_loc");
  let with_span = format_ident!("{ctor}_with_span");
  let is = format_ident!("is_{ctor}");
  let (arg, value, pattern, accessor) = if let Some(ty) = ty {
    let as_variant = format_ident!("as_{ctor}");
    (
      quote!(v: #ty,),
      quote!(#inner::#variant(v)),
      quote!(#inner::#variant(_)),
      quote! {
        pub fn #as_variant(&self) -> Option<&#ty> {
          match &self._node { #inner::#variant(v) => Some(v), _ => None }
        }
      },
    )
  } else {
    (quote!(), quote!(#inner::#variant), quote!(#inner::#variant), quote!())
  };
  quote! {
    pub fn #ctor(#arg) -> Self {
      Self { _loc: NodeLocation::default(), _node: #value }
    }
    pub fn #with_loc(#arg loc: NodeLocation) -> Self {
      Self { _loc: loc, _node: #value }
    }
    pub fn #with_span(#arg start: usize, end: usize) -> Self {
      Self { _loc: NodeLocation::from_span(start, end), _node: #value }
    }
    pub fn #is(&self) -> bool { matches!(&self._node, #pattern) }
    #accessor
  }
}
