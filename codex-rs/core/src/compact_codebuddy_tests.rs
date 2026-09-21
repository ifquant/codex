use super::*;
use pretty_assertions::assert_eq;
use serde_json::json;

fn envelope(value: serde_json::Value) -> ResponseItemEnvelope {
    ResponseItemEnvelope::new(serde_json::from_value(value).unwrap())
}

#[test]
fn retains_complete_recent_pairs_in_order_and_stops_at_oversized_output() {
    let call = envelope(
        json!({"type":"function_call","name":"read_file","call_id":"recent","arguments":"{}"}),
    );
    let output =
        envelope(json!({"type":"function_call_output","call_id":"recent","output":"evidence"}));
    let custom = envelope(
        json!({"type":"custom_tool_call","name":"apply_patch","call_id":"custom","input":"patch"}),
    );
    let custom_output =
        envelope(json!({"type":"custom_tool_call_output","call_id":"custom","output":"applied"}));
    let user = envelope(
        json!({"type":"message","role":"user","content":[{"type":"input_text","text":"current task"}]}),
    );
    let orphan =
        envelope(json!({"type":"function_call_output","call_id":"orphan","output":"unpaired"}));
    let expected = vec![
        call.clone(),
        output.clone(),
        custom.clone(),
        custom_output.clone(),
    ];
    let history = vec![
        call.clone(),
        output,
        user,
        call.clone(),
        expected[1].clone(),
        custom,
        custom_output,
        orphan,
    ];
    assert_eq!(recent_tool_exchanges(&history), expected);
    let oversized = envelope(
        json!({"type":"function_call_output","call_id":"recent","output":"x".repeat(8000)}),
    );
    assert_eq!(recent_tool_exchanges(&[call, oversized]), Vec::new());
    let mut many = Vec::new();
    for index in 0..20 {
        many.push(envelope(json!({"type":"function_call","name":"read_file","call_id":index.to_string(),"arguments":"{}"})));
        many.push(envelope(
            json!({"type":"function_call_output","call_id":index.to_string(),"output":"ok"}),
        ));
    }
    assert_eq!(recent_tool_exchanges(&many), many[24..]);
}
