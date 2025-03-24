export interface SourceData {
  id: string;
  [key: string]: any;
}

export interface SourceDataTranslations extends SourceData {
  locale: string;
  is_canonical: boolean;
  parent_id: string;
}

export interface MigratedData extends SourceData {
  last_modified: string;
  created: string;
  original_id: string;
  original_translations_id?: string; // Optional since address table doesn't have it
}
