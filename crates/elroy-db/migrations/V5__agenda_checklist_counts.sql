ALTER TABLE bootstrap_documents
    ADD COLUMN checklist_total INTEGER NOT NULL DEFAULT 0;

ALTER TABLE bootstrap_documents
    ADD COLUMN checklist_completed INTEGER NOT NULL DEFAULT 0;

ALTER TABLE agenda_items
    ADD COLUMN checklist_total INTEGER NOT NULL DEFAULT 0;

ALTER TABLE agenda_items
    ADD COLUMN checklist_completed INTEGER NOT NULL DEFAULT 0;
