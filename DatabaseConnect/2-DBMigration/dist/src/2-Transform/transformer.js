"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
exports.Transformer = void 0;
class Transformer {
    constructor(idConverter) {
        this.idConverter = idConverter;
    }
    transform(dataMap) {
        const transformedRecords = [];
        // Iterate through each entry in the data map
        dataMap.forEach(({ main, translations }, id) => {
            // Simply use the first translation if available
            const translation = translations.length > 0 ? translations[0] : null;
            // Transform the record using the abstract method that will be implemented by subclasses
            const transformedRecord = this.transformSingleRecord(main, translation);
            // Add the transformed record to our results array
            transformedRecords.push(transformedRecord);
        });
        return transformedRecords;
    }
}
exports.Transformer = Transformer;
