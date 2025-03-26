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
        let ids = [];
        // Get IDs for translation lookup
        for (let i = 0; i < mainRecords.length; i++) {
            const record = mainRecords[i];
            ids.push(record.ID);
        }
        console.log("These ids were mapped from mainRecords ", ids[0], ",", ids[1], "...");
        // Extract translation records for these IDs
        const translationRecords = await this.extractTranslationRecords(ids, locale);
        if (translationRecords.length > 0) {
            (0, exports.logJsonBlock)(`First record from ${this.sourceTables.translations}`, translationRecords[0]);
        }
        // Organize data into a map for easy access
        const dataMap = new Map();
        mainRecords.forEach((record) => {
            dataMap.set(record.ID, {
                main: record,
                translations: [],
            });
        });
        translationRecords.forEach((translation) => {
            // Ensure we're using the right field for joining
            const parentId = translation.PARENT_RECORD_ID;
            const entry = dataMap.get(parentId);
            if (entry) {
                entry.translations.push(translation);
            }
            else {
                console.log(`No matching record found for translation with parent ID: ${parentId}`);
            }
        });
        return dataMap;
    }
}
exports.Extractor = Extractor;
