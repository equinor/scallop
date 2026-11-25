use proc_macro::TokenStream;
use quote::quote;

use scallop_core::*;

#[proc_macro]
/// Compile a Scallop program from tokens or a Rust string literal.
pub fn scallop(tokens: TokenStream) -> TokenStream {
  let opt = compiler::CompileOptions::default();

  let compiled = if let Ok(source) = syn::parse::<syn::LitStr>(tokens.clone()) {
    compiler::compile_string_to_ram_with_options(source.value(), &opt)
  } else {
    let src = compiler::front::RustMacroSource::new(tokens.into());
    compiler::compile_source_to_ram_with_options(src, &opt)
  };
  let ram = match compiled {
    Ok(ram) => ram,
    Err(errs) => {
      let all_errs = errs.iter().map(|err| err.to_string()).collect::<Vec<_>>().join("\n");
      return quote! { compile_error!(#all_errs); }.into();
    }
  };

  // Generate related rust code
  let rs_mod = match ram.to_rs_module(&opt) {
    Ok(module) => module,
    Err(error) => {
      let message = error.to_string();
      return quote! { compile_error!(#message); }.into();
    }
  };
  let rs_create_edb_fn = ram.to_rs_create_edb_fn();

  quote! {
    #rs_mod
    #rs_create_edb_fn
  }
  .into()
}
