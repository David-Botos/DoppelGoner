export interface SourceData {
  ID: string;
  [key: string]: any;
}

export interface SourceDataTranslations extends SourceData {
  LOCALE: string;
  IS_CANONICAL: boolean;
  PARENT_RECORD_ID: string;
}

export interface MigratedData {
  id: string;
  last_modified: string;
  created: string;
  original_id: string;
  original_translations_id?: string | null; // Optional since address table doesn't have it
}
