"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
exports.Extractor = exports.logJsonBlock = void 0;
// Export the utility function for reuse elsewhere
const logJsonBlock = (label, data) => {
    console.log(`\n┌─────── ${label} ───────┐`);
    console.log(JSON.stringify(data, null, 2));
    console.log(`└${"─".repeat(label.length + 16)}┘\n`);
};
exports.logJsonBlock = logJsonBlock;
class Extractor {
    constructor(snowflakeClient, sourceTables) {
        this.snowflakeClient = snowflakeClient;
        this.sourceTables = sourceTables;
    }
    async extract(limit, offset, locale = "en") {
        // Extract main records
        const mainRecords = await this.extractMainRecords(limit, offset);
        if (mainRecords.length > 0) {
            (0, exports.logJsonBlock)(`First record from ${this.sourceTables.main}`, mainRecords[0]);
        }
        // Get IDs for translation lookup
        const ids = mainRecords.map((record) => record.id);
        // Extract translation records for these IDs
        const translationRecords = await this.extractTranslationRecords(ids, locale);
        if (translationRecords.length > 0) {
            (0, exports.logJsonBlock)(`First record from ${this.sourceTables.translations}`, translationRecords[0]);
        }
        // Organize data into a map for easy access
        const dataMap = new Map();
        mainRecords.forEach((record) => {
            dataMap.set(record.id, {
                main: record,
                translations: [],
            });
        });
        translationRecords.forEach((translation) => {
            const entry = dataMap.get(translation.PARENT_RECORD_ID);
            if (entry) {
                entry.translations.push(translation);
            }
        });
        return dataMap;
    }
}
exports.Extractor = Extractor;
