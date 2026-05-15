ALTER TABLE bootstrap_documents
    ADD COLUMN trigger_datetime TEXT;

ALTER TABLE bootstrap_documents
    ADD COLUMN trigger_context TEXT;

ALTER TABLE bootstrap_documents
    ADD COLUMN closing_comment TEXT;
