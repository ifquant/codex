use super::ContextualUserFragment;
use codex_protocol::models::ContentItemKind;

#[derive(Debug, Clone, PartialEq)]
pub(crate) struct OutputLimitRecovery;

impl ContextualUserFragment for OutputLimitRecovery {
    fn content_kind(&self) -> ContentItemKind {
        ContentItemKind("recovery.output_limit".to_string())
    }

    fn role(&self) -> &'static str {
        "user"
    }

    fn markers(&self) -> (&'static str, &'static str) {
        Self::type_markers()
    }

    fn type_markers() -> (&'static str, &'static str) {
        ("<output_limit_recovery>", "</output_limit_recovery>")
    }

    fn body(&self) -> String {
        "The previous response was truncated by the output limit. Its tool calls were discarded and were not executed. Continue the task from the existing context with shorter reasoning and smaller, complete tool calls. Split large writes or edits into bounded steps. Do not repeat completed work.".to_string()
    }
}
