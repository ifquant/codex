use super::*;
use pretty_assertions::assert_eq;

#[test]
fn folds_only_exact_repeated_large_results_and_keeps_a_verbatim_target() {
    let output = "same evidence\n".repeat(200);
    let mut input = Vec::new();
    for (id, name, arguments, text) in [
        ("old", "exec_command", "{\"cmd\":\"cat a\"}", output.clone()),
        (
            "different-command",
            "exec_command",
            "{\"cmd\":\"cat b\"}",
            output.clone(),
        ),
        (
            "different-tool",
            "read_file",
            "{\"cmd\":\"cat a\"}",
            output.clone(),
        ),
        (
            "changed",
            "exec_command",
            "{\"cmd\":\"cat a\"}",
            format!("{output}changed"),
        ),
        ("new", "exec_command", "{\"cmd\":\"cat a\"}", output.clone()),
    ] {
        input.push(json!({"type":"function_call","name":name,"call_id":id,"arguments":arguments}));
        input.push(json!({"type":"function_call_output","call_id":id,"output":text}));
    }
    let mut request =
        json!({"model":"deepseek-v4.1-flash","input":input,"reasoning":{"effort":"max"}});
    let original = request.clone();
    let (body, _) = encode(request.clone()).unwrap();
    assert_eq!(request, original);
    let messages = body["messages"].as_array().unwrap();
    assert_eq!(messages[1]["content"], output);
    let repeated = messages[9]["content"].as_str().unwrap();
    assert!(repeated.contains("tool_call_id=old"));
    assert!(repeated.len() < output.len());
    assert_eq!(messages[3]["content"], output);
    assert_eq!(messages[5]["content"], output);
    assert_eq!(messages[7]["content"], format!("{output}changed"));
    assert_eq!(body["reasoning_effort"], "max");
    assert!(
        serde_json::to_vec(&body).unwrap().len() < serde_json::to_vec(&original).unwrap().len()
    );
    let input = request["input"].as_array_mut().unwrap();
    input.push(json!({"type":"function_call","name":"exec_command","call_id":"third","arguments":"{\"cmd\":\"cat a\"}"}));
    input.push(json!({"type":"function_call_output","call_id":"third","output":output}));
    let (appended, _) = encode(request).unwrap();
    assert_eq!(
        &appended["messages"].as_array().unwrap()[..messages.len()],
        messages.as_slice()
    );
}

#[test]
fn repeated_small_results_remain_verbatim() {
    let request = json!({"model":"deepseek-v4.1-flash","input":[
        {"type":"function_call","name":"exec_command","call_id":"a","arguments":"{}"},
        {"type":"function_call_output","call_id":"a","output":"ok"},
        {"type":"function_call","name":"exec_command","call_id":"b","arguments":"{}"},
        {"type":"function_call_output","call_id":"b","output":"ok"}
    ]});
    let (body, _) = encode(request).unwrap();
    assert_eq!(body["messages"][1]["content"], "ok");
    assert_eq!(body["messages"][3]["content"], "ok");
}

#[test]
fn references_must_shrink_the_json_encoded_payload() {
    let escaped_id = "\n".repeat(600);
    let output = "x".repeat(1024);
    let request = json!({"model":"deepseek-v4.1-flash","input":[
        {"type":"function_call","name":"exec_command","call_id":escaped_id,"arguments":"{}"},
        {"type":"function_call_output","call_id":escaped_id,"output":output},
        {"type":"function_call","name":"exec_command","call_id":"new","arguments":"{}"},
        {"type":"function_call_output","call_id":"new","output":output}
    ]});
    let (body, _) = encode(request).unwrap();
    assert_eq!(body["messages"][1]["content"], output);
    assert_eq!(body["messages"][3]["content"], output);
}
