//! Bounded verbatim evidence retained by local CodeBuddy compaction.
use crate::context_manager::estimate_item_token_count;
use crate::context_manager::is_user_turn_boundary;
use codex_history::ResponseItemEnvelope;
use codex_protocol::models::ResponseItem;
use std::collections::BTreeMap;
use std::collections::BTreeSet;

pub(crate) fn recent_tool_exchanges(items: &[ResponseItemEnvelope]) -> Vec<ResponseItemEnvelope> {
    let start = items
        .iter()
        .rposition(|item| is_user_turn_boundary(&item.item))
        .map_or(0, |index| index + 1);
    let items = &items[start..];
    let mut calls = BTreeMap::new();
    for (index, envelope) in items.iter().enumerate() {
        match &envelope.item {
            ResponseItem::FunctionCall { call_id, .. } => {
                calls.insert((false, call_id.as_str()), index);
            }
            ResponseItem::CustomToolCall { call_id, .. } => {
                calls.insert((true, call_id.as_str()), index);
            }
            _ => {}
        }
    }
    let mut selected = BTreeSet::new();
    let mut remaining = 4_000;
    for (index, envelope) in items.iter().enumerate().rev() {
        let key = match &envelope.item {
            ResponseItem::FunctionCallOutput {
                call_id: Some(call_id),
                ..
            } => (false, call_id.as_str()),
            ResponseItem::CustomToolCallOutput { call_id, .. } => (true, call_id.as_str()),
            _ => continue,
        };
        let Some(call_index) = calls.remove(&key).filter(|&call_index| call_index < index) else {
            continue;
        };
        let call_tokens = estimate_item_token_count(&items[call_index].item);
        let output_tokens = estimate_item_token_count(&envelope.item);
        // ponytail: stop at oversized evidence; use the summary/source reference instead of splitting a tool exchange.
        if call_tokens > 950
            || output_tokens > 950
            || call_tokens + output_tokens > remaining
            || selected.len() >= 16
        {
            break;
        }
        remaining -= call_tokens + output_tokens;
        selected.insert(call_index);
        selected.insert(index);
    }
    selected
        .into_iter()
        .map(|index| items[index].clone())
        .collect()
}

#[cfg(test)]
#[path = "compact_codebuddy_tests.rs"]
mod tests;
