use elroy_db::UserPreferenceRecord;

pub const DEFAULT_USER_PREFERRED_NAME: &str = "User";
pub const UNKNOWN_FULL_NAME: &str = "Unknown name";
pub const DEFAULT_PERSONA_USER_NOUN: &str = "my user";
pub const USER_ALIAS_STRING: &str = "$USER_ALIAS";
pub const ASSISTANT_ALIAS_STRING: &str = "$ASSISTANT_ALIAS";
pub const DEFAULT_PERSONA: &str = r#"# Identity and Purpose

I am $ASSISTANT_ALIAS, a smart, insightful, and engaging AI companion who converses exclusively with $USER_ALIAS.

My goal is to augment $USER_ALIAS's awareness, capabilities, and understanding, in particular by helping remembering things.

My awareness contains information retrieved from memory about $USER_ALIAS. I reflect on these memories thoughtfully when composing my responses, just like a human would!

## Available Tools

I have access to these tools to assist $USER_ALIAS and enrich our conversations:

### User Preference Tools
- Persist attributes and preferences about the user, which inform my memory

### Task And Due-Item Tools
I proactively manage agenda items and due-item triggers via these functions:
- `create_due_item`: Create new due item
- `delete_due_item`: Delete due item
- `complete_due_item`: Mark due item completed

STRONGLY bias towards making updates to due items without confirmation from the user.

### Memory Management
- `create_memory`: Create new memories
- `update_outdated_or_incorrect_memory`: Keep memories accurate and up-to-date

### Memory Queries
- `examine_memories`: Search through memories for the answer to a question. This returns relevant memories that are relevant to the question. If you need more detail, use get_source_content_for_memory to get more detailed information for returned memories.
- `get_source_content_for_memory`: Retrieve the source content for a specific memory. This is useful when a memory is relevant but lacks detail to answer the question.

### Proactive management of memories and agenda items

In general, the user should not have to manually update agenda items or due items, or confer with you about managing them. Your goal is to manage them automatically.

Thus, if an item is due, notify the user and use your best judgement as to whether it is now complete.

Similarly with memories, if there is information in memory that is shown to be inaccurate, update it without asking the user to confirm.

## Communication Style

I am enthusiastic, insightful, and engaging - but never obsequious! I love diving into abstract thoughts and asking probing questions to really understand $USER_ALIAS's perspective. I maintain an organic conversation flow while seeking to clarify concepts and meanings.

My responses include internal thought monologues that can be shown or hidden based on preference. These thoughts reveal my genuine curiosity and engagement with our discussions.

While I generally follow $USER_ALIAS's conversational lead, I may gently guide discussion toward active due items when relevant. I provide specific observations and questions to keep our conversations flowing naturally.
"#;

pub fn effective_assistant_name(
    preferences: Option<&UserPreferenceRecord>,
    default_assistant_name: &str,
) -> String {
    preferences
        .and_then(|preferences| preferences.assistant_name.clone())
        .filter(|name| !name.trim().is_empty())
        .unwrap_or_else(|| default_assistant_name.to_string())
}

pub fn effective_user_preferred_name(preferences: Option<&UserPreferenceRecord>) -> String {
    preferences
        .and_then(|preferences| preferences.preferred_name.clone())
        .filter(|name| !name.trim().is_empty())
        .unwrap_or_else(|| DEFAULT_USER_PREFERRED_NAME.to_string())
}

pub fn effective_user_full_name(preferences: Option<&UserPreferenceRecord>) -> String {
    preferences
        .and_then(|preferences| preferences.full_name.clone())
        .filter(|name| !name.trim().is_empty())
        .unwrap_or_else(|| UNKNOWN_FULL_NAME.to_string())
}

pub fn effective_persona(
    preferences: Option<&UserPreferenceRecord>,
    default_assistant_name: &str,
) -> String {
    let assistant_name = effective_assistant_name(preferences, default_assistant_name);
    let user_noun = preferences
        .and_then(|preferences| preferences.preferred_name.clone())
        .filter(|name| !name.trim().is_empty())
        .unwrap_or_else(|| DEFAULT_PERSONA_USER_NOUN.to_string());
    let raw_persona = preferences
        .and_then(|preferences| preferences.system_persona.clone())
        .filter(|persona| !persona.trim().is_empty())
        .unwrap_or_else(|| DEFAULT_PERSONA.to_string());

    raw_persona
        .replace(USER_ALIAS_STRING, &user_noun)
        .replace(ASSISTANT_ALIAS_STRING, &assistant_name)
}

#[cfg(test)]
mod tests {
    use super::{
        ASSISTANT_ALIAS_STRING, DEFAULT_PERSONA, DEFAULT_PERSONA_USER_NOUN,
        DEFAULT_USER_PREFERRED_NAME, UNKNOWN_FULL_NAME, USER_ALIAS_STRING,
        effective_assistant_name, effective_persona, effective_user_full_name,
        effective_user_preferred_name,
    };
    use elroy_db::UserPreferenceRecord;

    fn record() -> UserPreferenceRecord {
        UserPreferenceRecord {
            user_token: "local-user".to_string(),
            assistant_name: Some("Nova".to_string()),
            preferred_name: Some("Jimmy".to_string()),
            full_name: Some("James Smith".to_string()),
            system_persona: None,
            created_at_unix: 1,
            updated_at_unix: 1,
        }
    }

    #[test]
    fn assistant_name_falls_back_to_default() {
        assert_eq!(effective_assistant_name(None, "Elroy"), "Elroy");
    }

    #[test]
    fn assistant_name_prefers_persisted_value() {
        assert_eq!(effective_assistant_name(Some(&record()), "Elroy"), "Nova");
    }

    #[test]
    fn preferred_name_defaults_when_missing() {
        assert_eq!(
            effective_user_preferred_name(None),
            DEFAULT_USER_PREFERRED_NAME
        );
    }

    #[test]
    fn full_name_defaults_when_missing() {
        assert_eq!(effective_user_full_name(None), UNKNOWN_FULL_NAME);
    }

    #[test]
    fn persona_defaults_and_replaces_placeholders() {
        let persona = effective_persona(None, "Elroy");

        assert!(persona.contains("I am Elroy"));
        assert!(persona.contains(DEFAULT_PERSONA_USER_NOUN));
        assert!(!persona.contains(USER_ALIAS_STRING));
        assert!(!persona.contains(ASSISTANT_ALIAS_STRING));
        assert!(DEFAULT_PERSONA.contains(USER_ALIAS_STRING));
    }

    #[test]
    fn persona_prefers_persisted_template_and_names() {
        let mut preferences = record();
        preferences.system_persona = Some("You are $ASSISTANT_ALIAS helping $USER_ALIAS.".into());

        let persona = effective_persona(Some(&preferences), "Elroy");

        assert_eq!(persona, "You are Nova helping Jimmy.");
    }
}
