# Human Services Data Specification (HSDS) Schema

The Human Services Data Specification (HSDS) is a data model that defines a collection of objects and their relationships for representing information about human services, the organizations that provide them, and the locations where they're offered.

## Overview

HSDS data can be serialized as a Tabular Data Package consisting of CSV files (one for each object) and a package descriptor file (`datapackage.json`).

## Core Objects and Fields

### Organization

The organization record provides basic description and details about each organization delivering services.

| Field Name | Type (Format) | Description | Required? | Unique? |
|------------|---------------|-------------|-----------|---------|
| id | string (uuid) | Each organization must have a unique identifier | True | True |
| name | string | The official or public name of the organization | True | False |
| alternate_name | string | Alternative or commonly used name for the organization | False | False |
| description | string | A brief summary about the organization. It can contain markup such as HTML or Markdown | True | False |
| email | string (email) | The contact e-mail address for the organization | False | False |
| url | string (uri) | The URL (website address) of the organization | False | False |
| tax_status | string | Government assigned tax designation for tax-exempt organizations | False | False |
| tax_id | string | A government issued identifier used for the purpose of tax administration | False | False |
| year_incorporated | date (%Y) | The year in which the organization was legally formed | False | False |
| legal_status | string | The legal status defines the conditions that an organization is operating under | False | False |

### Program

Programs bring together a number of related services within an organization.

| Field Name | Type (Format) | Description | Required? | Unique? |
|------------|---------------|-------------|-----------|---------|
| id | string | Each program must have a unique identifier | True | True |
| organization_id | string (uuid) | Each program must belong to a single organization | True | True |
| name | string | The name of the program | True | False |
| alternate_name | string | An alternative name for the program | False | False |

### Service

Services are provided by organizations to different groups.

| Field Name | Type (Format) | Description | Required? | Unique? |
|------------|---------------|-------------|-----------|---------|
| id | string | Each service must have a unique identifier | True | True |
| organization_id | string | The identifier of the organization that provides this service | True | False |
| program_id | string | The identifier of the program this service is delivered under | False | False |
| name | string | The official or public name of the service | True | False |
| alternate_name | string | Alternative or commonly used name for a service | False | False |
| description | string | A description of the service | False | False |
| url | string (uri) | URL of the service | False | False |
| email | string (email) | Email address for the service | False | False |
| status | string | The current status of the service | True | False |
| interpretation_services | string | A description of any interpretation services available | False | False |
| application_process | string | The steps needed to access the service | False | False |
| wait_time | string | Time a client may expect to wait before receiving a service | False | False |
| fees | string | Details of any charges for service users | False | False |
| accreditations | string | Details of any accreditations | False | False |
| licenses | string | Any licenses issued by a government entity | False | False |

### Location

The location tables provides details of the locations where organizations operate.

| Field Name | Type (Format) | Description | Required? | Unique? |
|------------|---------------|-------------|-----------|---------|
| id | string | Each location must have a unique identifier | True | True |
| organization_id | string | Linked to a single organization responsible for this location | False | False |
| name | string | The name of the location | False | False |
| alternate_name | string | An alternative name for the location | False | False |
| description | string | A description of this location | False | False |
| transportation | string | A description of the access to public or private transportation | False | False |
| latitude | number | Y coordinate in decimal degrees (WGS84 datum) | False | False |
| longitude | number | X coordinate in decimal degrees (WGS84 datum) | False | False |

### Service At Location

Creates a link between a service and a specific location.

| Field Name | Type (Format) | Description | Required? | Unique? |
|------------|---------------|-------------|-----------|---------|
| id | string | Each entry must have a unique identifier | True | True |
| service_id | string | The identifier of the service at a given location | True | False |
| location_id | string | The identifier of the location where this service operates | True | False |
| description | string | Additional information about the service at this location | False | False |

### Taxonomy Term

Used to categorize services according to one or more taxonomy terms.

| Field Name | Type (Format) | Description | Required? | Unique? |
|------------|---------------|-------------|-----------|---------|
| id | string | Each taxonomy term must have a unique identifier | True | True |
| term | string | The taxonomy term itself | True | False |
| description | string | What the term means | True | False |
| parent_id | string | Identifier of the parent category (for hierarchical taxonomies) | False | False |
| taxonomy | string | Which established taxonomy is in use (e.g., AIRS) | False | False |
| language | string | ISO language code for the language of the term | False | False |

### Service Attribute

Links a service to one or more classifications that describe its nature.

| Field Name | Type (Format) | Description | Required? | Unique? |
|------------|---------------|-------------|-----------|---------|
| id | string | Each service_attribute entry should have a unique identifier | True | True |
| service_id | string | The identifier of the service being classified | True | False |
| taxonomy_term_id | string | The identifier of the taxonomy term that applies | False | False |

### Other Attribute

Links entities other than services to classifications.

| Field Name | Type (Format) | Description | Required? | Unique? |
|------------|---------------|-------------|-----------|---------|
| id | string | Each other_attribute entry should have a unique identifier | True | True |
| link_id | string | The identifier of the entity being classified | True | False |
| link_type | string | The entity type being classified | True | False |
| taxonomy_term_id | string | The identifier of the taxonomy term that applies | False | False |

### Contact

Contains details of named contacts for services and organizations.

| Field Name | Type (Format) | Description | Required? | Unique? |
|------------|---------------|-------------|-----------|---------|
| id | string | Each contact must have a unique identifier | True | True |
| organization_id | string | The organization for which this is a contact | False | False |
| service_id | string | The service for which this is a contact | False | False |
| service_at_location_id | string | When contact is specific to a service at a location | False | False |
| name | string | The name of the person | False | False |
| title | string | The job title of the person | False | False |
| department | string | The department that the person is part of | False | False |
| email | string (email) | The email address of the person | False | False |

### Phone

Contains telephone numbers used to contact organizations, services, and locations.

| Field Name | Type (Format) | Description | Required? | Unique? |
|------------|---------------|-------------|-----------|---------|
| id | string | Each entry must have a unique identifier | True | True |
| location_id | string | The location where this phone number is located | False | False |
| service_id | string | The service for which this is the phone number | False | False |
| organization_id | string | The organization for which this is the phone number | False | False |
| contact_id | string | The contact for which this is the phone number | False | False |
| service_at_location_id | string | When phone is specific to a service at a location | False | False |
| number | string | The phone number | True | False |
| extension | number | The extension of the phone number | False | False |
| type | string | Type of phone service (voice, fax, cell, etc.) | False | False |
| language | string | ISO language codes for available languages | False | False |
| description | string | Extra information about the phone service | False | False |

### Physical Address

Contains the physical addresses for locations.

| Field Name | Type (Format) | Description | Required? | Unique? |
|------------|---------------|-------------|-----------|---------|
| id | string | Each physical address must have a unique identifier | True | True |
| location_id | string | The location for which this is the address | False | False |
| attention | string | The person or entity whose attention should be sought | False | False |
| address_1 | string | First line(s) of the address | True | False |
| city | string | The city in which the address is located | True | False |
| region | string | The region in which the address is located | False | False |
| state_province | string | The state or province in which the address is located | True | False |
| postal_code | string | The postal code for the address | True | False |
| country | string | ISO 3361-1 country code | True | False |

### Postal Address

Contains postal addresses for mail to a location (may differ from physical location).

| Field Name | Type (Format) | Description | Required? | Unique? |
|------------|---------------|-------------|-----------|---------|
| id | string | Each postal address must have a unique identifier | True | True |
| location_id | string | The location for which this is the postal address | False | False |
| attention | string | The person or entity whose attention should be sought | False | False |
| address_1 | string | First line(s) of the address | True | False |
| city | string | The city in which the address is located | True | False |
| region | string | The region in which the address is located | False | False |
| state_province | string | The state or province in which the address is located | True | False |
| postal_code | string | The postal code for the address | True | False |
| country | string | ISO 3361-1 country code | True | False |

### Schedule

Contains details of when a service or location is open using RFC 5545 RRULES.

| Field Name | Type (Format) | Description | Required? | Unique? |
|------------|---------------|-------------|-----------|---------|
| id | string | Each entry must have a unique identifier | True | True |
| service_id | string | The service for which this is the schedule | False | False |
| location_id | string | The location for which this is the schedule | False | False |
| service_at_location_id | string | When schedule is specific to a service at a location | False | False |
| valid_from | date | Date from which the schedule information is valid | False | False |
| valid_to | date | Last date on which the schedule information is valid | False | False |
| dtstart | date | The date of the first event in the schedule | False | False |
| freq | string | How often the frequency repeats (WEEKLY/MONTHLY) | False | False |
| interval | number | How often the frequency repeats (e.g., 2 = biweekly) | False | False |
| byday | string | Comma separated days of the week | False | False |
| description | string | Human readable description of availability | False | False |

### Additional Objects

#### Funding

| Field Name | Type (Format) | Description | Required? | Unique? |
|------------|---------------|-------------|-----------|---------|
| id | string | Each entry must have a unique identifier | True | True |
| organization_id | string | The organization in receipt of this funding | False | False |
| service_id | string | The service in receipt of this funding | False | False |
| source | string | Description of the source of funds | False | False |

#### Service Area

| Field Name | Type (Format) | Description | Required? | Unique? |
|------------|---------------|-------------|-----------|---------|
| id | string | Each service area must have a unique identifier | True | True |
| service_id | string | The service for which this entry describes the area | False | False |
| service_area | string | Free-text description of the geographic area | False | False |
| description | string | More detailed description of this service area | False | False |

#### Required Document

| Field Name | Type (Format) | Description | Required? | Unique? |
|------------|---------------|-------------|-----------|---------|
| id | string | Each document must have a unique identifier | True | True |
| service_id | string | The service requiring this document | False | False |
| document | string | The document required to apply for or receive the service | False | False |

#### Eligibility

| Field Name | Type (Format) | Description | Required? | Unique? |
|------------|---------------|-------------|-----------|---------|
| id | string | Each entry must have a unique identifier | True | True |
| service_id | string | The service with these eligibility criteria | False | False |

#### Payment Accepted

| Field Name | Type (Format) | Description | Required? | Unique? |
|------------|---------------|-------------|-----------|---------|
| id | string | Each entry must have a unique identifier | True | True |
| service_id | string | The service with these payment methods | False | False |
| payment | string | The methods of payment accepted for the service | False | False |

#### Language

| Field Name | Type (Format) | Description | Required? | Unique? |
|------------|---------------|-------------|-----------|---------|
| id | string | Each language must have a unique identifier | True | True |
| service_id | string | The service with these available languages | False | False |
| location_id | string | The location with these available languages | False | False |
| language | string | ISO639-1 codes for languages other than English | False | False |

#### Accessibility For Disabilities

| Field Name | Type (Format) | Description | Required? | Unique? |
|------------|---------------|-------------|-----------|---------|
| id | string | Each entry must have a unique identifier | True | True |
| location_id | string | The location with these accessibility provisions | False | False |
| accessibility | string | Description of assistance or infrastructure | False | False |
| details | string | Further details about accessibility arrangements | False | False |

#### Metadata

| Field Name | Type (Format) | Description | Required? | Unique? |
|------------|---------------|-------------|-----------|---------|
| id | string | Each entry must have a unique identifier | True | True |
| resource_id | string | Identifier of the entity being referenced | True | True |
| resource_type | string | The type of entity being referenced | True | True |
| last_action_date | datetime | The date when data was changed | True | False |
| last_action_type | string | The kind of change made to the data | True | False |
| field_name | string | The name of field that has been modified | True | False |
| previous_value | string | The previous value of a field that has been updated | True | False |
| replacement_value | string | The new value of a field that has been updated | True | False |
| updated_by | string | The name of the person who updated a value | True | False |

## Relationships

The HSDS schema establishes the following key relationships:

1. Organizations provide Services
2. Services may be grouped into Programs
3. Services are offered at Locations (via the Service_At_Location table)
4. Services are categorized by Taxonomy_Terms (via the Service_Attribute table)
5. Locations have Physical_Addresses and may have Postal_Addresses
6. Services, Locations, and Organizations may have Contact information
7. Services, Locations, and Organizations may have Phone numbers
8. Services and Locations have Schedules for availability
9. Services have Service_Areas defining their geographic coverage
10. Services have Eligibility requirements and may need Required_Documents
11. Services accept certain Payment methods
12. Locations have Accessibility_For_Disabilities information

## Format Conventions

- **Name and Alternate_Name fields**: Should only include plain text without formatting
- **Description fields**: May include HTML elements or Markdown, with special characters escaped or encoded
- **Dates**: Follow ISO format standards
- **Language codes**: Use ISO 639-1 or ISO 639-2 codes
- **Country codes**: Use ISO 3361-1 country codes (two letter abbreviations)
