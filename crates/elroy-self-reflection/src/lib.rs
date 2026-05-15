use std::path::Path;

use chrono::Utc;
use elroy_feature_requests::{
    FeatureRequestMatch, FeatureRequestRecord, find_best_feature_request_match,
    is_active_feature_request, update_feature_request, write_new_feature_request,
};
use elroy_llm::{ConversationMessage, MessageRole};

const CORRECTION_PHRASES: &[&str] = &[
    "you forgot",
    "that's wrong",
    "not quite",
    "reflect on that",
    "you should improve",
];

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SelfReflectionConfig {
    pub messages_between_self_reflection: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ReflectionProposal {
    pub title: String,
    pub description: String,
    pub rationale: String,
    pub supporting_context: String,
    pub feedback_excerpt: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SelfReflectionResult {
    pub triggered: bool,
    pub proposal: Option<ReflectionProposal>,
    pub feature_request: Option<FeatureRequestRecord>,
}

pub struct SelfReflectionOrchestrator {
    config: SelfReflectionConfig,
}

impl SelfReflectionOrchestrator {
    pub fn new(config: SelfReflectionConfig) -> Self {
        Self { config }
    }

    pub fn run(
        &self,
        home_dir: &Path,
        context_messages: &[ConversationMessage],
    ) -> std::io::Result<SelfReflectionResult> {
        if !self.should_reflect(context_messages) {
            return Ok(SelfReflectionResult {
                triggered: false,
                proposal: None,
                feature_request: None,
            });
        }

        let Some(proposal) = self.build_proposal(context_messages) else {
            return Ok(SelfReflectionResult {
                triggered: false,
                proposal: None,
                feature_request: None,
            });
        };

        let feature_request = self.persist_feature_request(home_dir, &proposal)?;
        Ok(SelfReflectionResult {
            triggered: true,
            proposal: Some(proposal),
            feature_request: Some(feature_request),
        })
    }

    fn should_reflect(&self, context_messages: &[ConversationMessage]) -> bool {
        let threshold = self.config.messages_between_self_reflection;
        if threshold == 0 {
            return false;
        }

        let user_message_count = context_messages
            .iter()
            .filter(|message| {
                message.role == MessageRole::User
                    && message
                        .content
                        .as_deref()
                        .is_some_and(|content| !content.trim().is_empty())
            })
            .count();
        user_message_count >= threshold && user_message_count % threshold == 0
    }

    fn build_proposal(
        &self,
        context_messages: &[ConversationMessage],
    ) -> Option<ReflectionProposal> {
        for message in context_messages.iter().rev() {
            if message.role != MessageRole::User {
                continue;
            }
            let content = message.content.as_deref()?.trim();
            if content.is_empty() {
                continue;
            }

            let normalized = content.to_ascii_lowercase();
            let phrase = CORRECTION_PHRASES
                .iter()
                .copied()
                .find(|candidate| normalized.contains(candidate))?;

            let mut excerpt = content.split_whitespace().collect::<Vec<_>>().join(" ");
            if excerpt.chars().count() > 240 {
                excerpt = format!(
                    "{}...",
                    excerpt.chars().take(237).collect::<String>().trim_end()
                );
            }

            return Some(ReflectionProposal {
                title: "Improve response handling after direct user corrections".to_string(),
                description: "When the user gives explicit correction-like feedback, Elroy should treat it as a signal to tighten response validation and recover more directly within the same conversation turn.".to_string(),
                rationale: format!(
                    "Recent feedback included the phrase '{phrase}', which suggests the assistant gave an incomplete or incorrect response and should adapt more reliably."
                ),
                supporting_context: [
                    format!("- Reflected at: {}", Utc::now().to_rfc3339()),
                    format!("- Trigger phrase: {phrase}"),
                    format!("- Recent user feedback: {excerpt}"),
                ]
                .join("\n"),
                feedback_excerpt: excerpt,
            });
        }

        None
    }

    fn persist_feature_request(
        &self,
        home_dir: &Path,
        proposal: &ReflectionProposal,
    ) -> std::io::Result<FeatureRequestRecord> {
        if let Some(matched) =
            find_best_feature_request_match(home_dir, &proposal.title, &proposal.description)?
        {
            return self.update_matched_request(matched, proposal);
        }

        write_new_feature_request(
            home_dir,
            &proposal.title,
            &proposal.description,
            Some(&proposal.rationale),
            Some(&proposal.supporting_context),
            "self_reflection",
        )
    }

    fn update_matched_request(
        &self,
        matched: FeatureRequestMatch,
        proposal: &ReflectionProposal,
    ) -> std::io::Result<FeatureRequestRecord> {
        let reopened_status = if matched.record.source == "self_reflection"
            && !is_active_feature_request(&matched.record)
        {
            Some("open")
        } else {
            None
        };

        if has_matching_feedback_excerpt(
            matched.record.supporting_context.as_deref(),
            &proposal.feedback_excerpt,
        ) {
            if let Some(status) = reopened_status {
                return update_feature_request(
                    &matched.record,
                    None,
                    Some(status),
                    None,
                    None,
                    None,
                    None,
                );
            }
            return Ok(matched.record);
        }

        let mut aliases = matched.record.aliases.clone();
        if proposal.title != matched.record.title
            && !aliases.iter().any(|alias| alias == &proposal.title)
        {
            aliases.push(proposal.title.clone());
            aliases.sort();
        }

        update_feature_request(
            &matched.record,
            None,
            reopened_status,
            Some(&aliases),
            None,
            None,
            Some(Some(&merge_supporting_context(
                matched.record.supporting_context.as_deref(),
                &proposal.supporting_context,
            ))),
        )
    }
}

fn merge_supporting_context(existing: Option<&str>, new_context: &str) -> String {
    match existing {
        None => new_context.to_string(),
        Some(existing) if existing.contains(new_context) => existing.to_string(),
        Some(existing) => format!("{}\n\n{}", existing.trim_end(), new_context),
    }
}

fn has_matching_feedback_excerpt(existing: Option<&str>, feedback_excerpt: &str) -> bool {
    existing.is_some_and(|existing| {
        existing.contains(&format!("- Recent user feedback: {feedback_excerpt}"))
    })
}

#[cfg(test)]
mod tests {
    use super::{SelfReflectionConfig, SelfReflectionOrchestrator};
    use elroy_feature_requests::{list_feature_requests, update_feature_request};
    use elroy_llm::{ConversationMessage, MessageRole};

    fn unique_home(prefix: &str) -> std::path::PathBuf {
        let unique = format!(
            "{prefix}-{}",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .expect("system clock should be after unix epoch")
                .as_nanos()
        );
        std::env::temp_dir().join(unique)
    }

    #[test]
    fn no_feature_request_without_correction_feedback() {
        let home = unique_home("elroy-rs-self-reflection-none");
        std::fs::create_dir_all(&home).expect("home should exist");
        let orchestrator = SelfReflectionOrchestrator::new(SelfReflectionConfig {
            messages_between_self_reflection: 2,
        });

        let result = orchestrator
            .run(
                &home,
                &[
                    ConversationMessage::new(MessageRole::User, "Can you summarize the plan?"),
                    ConversationMessage::new(MessageRole::Assistant, "Here is the summary."),
                    ConversationMessage::new(MessageRole::User, "Thanks, now make it shorter."),
                    ConversationMessage::new(MessageRole::Assistant, "Shorter version."),
                ],
            )
            .expect("reflection should run");

        assert!(!result.triggered);
        assert!(
            list_feature_requests(&home)
                .expect("feature requests should load")
                .is_empty()
        );
        std::fs::remove_dir_all(home).expect("home should be removed");
    }

    #[test]
    fn creates_feature_request_when_threshold_and_correction_hit() {
        let home = unique_home("elroy-rs-self-reflection-create");
        std::fs::create_dir_all(&home).expect("home should exist");
        let orchestrator = SelfReflectionOrchestrator::new(SelfReflectionConfig {
            messages_between_self_reflection: 2,
        });

        let result = orchestrator
            .run(
                &home,
                &[
                    ConversationMessage::new(MessageRole::User, "Draft a reply to this message."),
                    ConversationMessage::new(MessageRole::Assistant, "Here is a draft."),
                    ConversationMessage::new(
                        MessageRole::User,
                        "That's wrong. You forgot the main deadline.",
                    ),
                    ConversationMessage::new(MessageRole::Assistant, "I will revise it."),
                ],
            )
            .expect("reflection should run");

        let records = list_feature_requests(&home).expect("feature requests should load");
        assert!(result.triggered);
        assert_eq!(records.len(), 1);
        assert_eq!(
            records[0].title,
            "Improve response handling after direct user corrections"
        );
        assert_eq!(records[0].source, "self_reflection");
        assert!(records[0].summary.contains("tighten response validation"));
        assert!(
            records[0]
                .rationale
                .as_deref()
                .is_some_and(|value| value.to_ascii_lowercase().contains("you forgot"))
        );
        assert!(
            records[0]
                .supporting_context
                .as_deref()
                .is_some_and(|value| value.contains("You forgot the main deadline."))
        );
        std::fs::remove_dir_all(home).expect("home should be removed");
    }

    #[test]
    fn skips_when_cadence_threshold_not_hit() {
        let home = unique_home("elroy-rs-self-reflection-threshold");
        std::fs::create_dir_all(&home).expect("home should exist");
        let orchestrator = SelfReflectionOrchestrator::new(SelfReflectionConfig {
            messages_between_self_reflection: 2,
        });

        let result = orchestrator
            .run(
                &home,
                &[
                    ConversationMessage::new(
                        MessageRole::User,
                        "That's wrong, you forgot the dependency note.",
                    ),
                    ConversationMessage::new(MessageRole::Assistant, "I will fix it."),
                ],
            )
            .expect("reflection should run");

        assert!(!result.triggered);
        assert!(
            list_feature_requests(&home)
                .expect("feature requests should load")
                .is_empty()
        );
        std::fs::remove_dir_all(home).expect("home should be removed");
    }

    #[test]
    fn dedupes_repeated_triggers_and_merges_context() {
        let home = unique_home("elroy-rs-self-reflection-dedupe");
        std::fs::create_dir_all(&home).expect("home should exist");
        let orchestrator = SelfReflectionOrchestrator::new(SelfReflectionConfig {
            messages_between_self_reflection: 2,
        });

        orchestrator
            .run(
                &home,
                &[
                    ConversationMessage::new(MessageRole::User, "Give me the release checklist."),
                    ConversationMessage::new(MessageRole::Assistant, "Checklist draft."),
                    ConversationMessage::new(
                        MessageRole::User,
                        "Not quite, you forgot the rollback step.",
                    ),
                    ConversationMessage::new(MessageRole::Assistant, "I will add it."),
                ],
            )
            .expect("first reflection should run");
        orchestrator
            .run(
                &home,
                &[
                    ConversationMessage::new(MessageRole::User, "Give me the release checklist."),
                    ConversationMessage::new(MessageRole::Assistant, "Checklist draft."),
                    ConversationMessage::new(
                        MessageRole::User,
                        "Not quite, you forgot the rollback step.",
                    ),
                    ConversationMessage::new(MessageRole::Assistant, "I will add it."),
                    ConversationMessage::new(MessageRole::User, "Now shorten it."),
                    ConversationMessage::new(MessageRole::Assistant, "Short version."),
                    ConversationMessage::new(
                        MessageRole::User,
                        "You should improve how you handle corrections like this.",
                    ),
                    ConversationMessage::new(MessageRole::Assistant, "Understood."),
                ],
            )
            .expect("second reflection should run");

        let records = list_feature_requests(&home).expect("feature requests should load");
        assert_eq!(records.len(), 1);
        let supporting_context = records[0].supporting_context.as_deref().unwrap_or_default();
        assert!(supporting_context.contains("rollback step"));
        assert!(supporting_context.contains("handle corrections like this"));
        std::fs::remove_dir_all(home).expect("home should be removed");
    }

    #[test]
    fn does_not_repeat_same_feedback_on_later_thresholds() {
        let home = unique_home("elroy-rs-self-reflection-repeat");
        std::fs::create_dir_all(&home).expect("home should exist");
        let orchestrator = SelfReflectionOrchestrator::new(SelfReflectionConfig {
            messages_between_self_reflection: 2,
        });

        orchestrator
            .run(
                &home,
                &[
                    ConversationMessage::new(MessageRole::User, "Show me the release steps."),
                    ConversationMessage::new(MessageRole::Assistant, "Here are the steps."),
                    ConversationMessage::new(
                        MessageRole::User,
                        "Not quite, you forgot the rollback step.",
                    ),
                    ConversationMessage::new(MessageRole::Assistant, "I will revise it."),
                ],
            )
            .expect("first reflection should run");
        let first_supporting_context = list_feature_requests(&home)
            .expect("feature requests should load")[0]
            .supporting_context
            .clone();

        orchestrator
            .run(
                &home,
                &[
                    ConversationMessage::new(MessageRole::User, "Show me the release steps."),
                    ConversationMessage::new(MessageRole::Assistant, "Here are the steps."),
                    ConversationMessage::new(
                        MessageRole::User,
                        "Not quite, you forgot the rollback step.",
                    ),
                    ConversationMessage::new(MessageRole::Assistant, "I will revise it."),
                    ConversationMessage::new(MessageRole::User, "Shorten the checklist."),
                    ConversationMessage::new(MessageRole::Assistant, "Shorter checklist."),
                    ConversationMessage::new(MessageRole::User, "Make it one sentence."),
                    ConversationMessage::new(MessageRole::Assistant, "One sentence version."),
                ],
            )
            .expect("second reflection should run");

        let records = list_feature_requests(&home).expect("feature requests should load");
        assert_eq!(records.len(), 1);
        assert_eq!(records[0].supporting_context, first_supporting_context);
        std::fs::remove_dir_all(home).expect("home should be removed");
    }

    #[test]
    fn can_be_disabled_with_zero() {
        let home = unique_home("elroy-rs-self-reflection-disabled");
        std::fs::create_dir_all(&home).expect("home should exist");
        let orchestrator = SelfReflectionOrchestrator::new(SelfReflectionConfig {
            messages_between_self_reflection: 0,
        });

        let result = orchestrator
            .run(
                &home,
                &[
                    ConversationMessage::new(MessageRole::User, "Draft a reply to this message."),
                    ConversationMessage::new(MessageRole::Assistant, "Here is a draft."),
                    ConversationMessage::new(
                        MessageRole::User,
                        "That's wrong. You forgot the main deadline.",
                    ),
                    ConversationMessage::new(MessageRole::Assistant, "I will revise it."),
                ],
            )
            .expect("reflection should run");

        assert!(!result.triggered);
        assert!(
            list_feature_requests(&home)
                .expect("feature requests should load")
                .is_empty()
        );
        std::fs::remove_dir_all(home).expect("home should be removed");
    }

    #[test]
    fn reopens_closed_matching_feature_request() {
        let home = unique_home("elroy-rs-self-reflection-reopen");
        std::fs::create_dir_all(&home).expect("home should exist");
        let orchestrator = SelfReflectionOrchestrator::new(SelfReflectionConfig {
            messages_between_self_reflection: 2,
        });

        orchestrator
            .run(
                &home,
                &[
                    ConversationMessage::new(MessageRole::User, "Give me the release checklist."),
                    ConversationMessage::new(MessageRole::Assistant, "Checklist draft."),
                    ConversationMessage::new(
                        MessageRole::User,
                        "Not quite, you forgot the rollback step.",
                    ),
                    ConversationMessage::new(MessageRole::Assistant, "I will add it."),
                ],
            )
            .expect("first reflection should run");

        let closed = update_feature_request(
            &list_feature_requests(&home).expect("feature requests should load")[0],
            None,
            Some("closed"),
            None,
            None,
            None,
            None,
        )
        .expect("feature request should close");
        assert_eq!(closed.status, "closed");

        orchestrator
            .run(
                &home,
                &[
                    ConversationMessage::new(MessageRole::User, "Give me the release checklist."),
                    ConversationMessage::new(MessageRole::Assistant, "Checklist draft."),
                    ConversationMessage::new(
                        MessageRole::User,
                        "Not quite, you forgot the rollback step.",
                    ),
                    ConversationMessage::new(MessageRole::Assistant, "I will add it."),
                    ConversationMessage::new(MessageRole::User, "Now shorten it."),
                    ConversationMessage::new(MessageRole::Assistant, "Short version."),
                    ConversationMessage::new(
                        MessageRole::User,
                        "You should improve how you handle corrections like this.",
                    ),
                    ConversationMessage::new(MessageRole::Assistant, "Understood."),
                ],
            )
            .expect("second reflection should run");

        let records = list_feature_requests(&home).expect("feature requests should load");
        assert_eq!(records.len(), 1);
        assert_eq!(records[0].status, "open");
        let supporting_context = records[0].supporting_context.as_deref().unwrap_or_default();
        assert!(supporting_context.contains("rollback step"));
        assert!(supporting_context.contains("handle corrections like this"));
        std::fs::remove_dir_all(home).expect("home should be removed");
    }
}
