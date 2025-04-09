# Social Services Database Tables

## ADDRESS TABLE

| id | city | region | country | address_1 | address_2 | attention | location_id | postal_code | state_province |
|----|------|--------|---------|-----------|-----------|-----------|-------------|-------------|----------------|
| 723d0492-eebf-4751-bf3e-ce5cd7250620 | Sedro-Woolley | Skagit | US | 2060 Powerhouse Drive | null | null | 783b945d-6fe8-4426-970d-89689d7849b1 | 98284 | WA |
| a3d14fb8-bfb6-4b8b-8143-8d5d6bdb9da0 | Montesano | Grays Harbor | US | 112 North Main Street | null | null | 45e635ac-539a-499b-a5d4-532074c3f49b | 98563 | WA |
| 8207ba4c-907e-42a0-a2e5-2be91a510bc1 | Seattle | King | US | 600 Pine Street | Suite 310 | null | daf95de0-9add-494c-8397-72c03c28e8b4 | 98101 | WA |
| 6d895586-348e-4b6a-a29b-904cdbb4c3ca | Port Angeles | Clallam | US | 132 East 13th | null | null | f3b90ec1-f514-4e1c-94de-ef4d9fa9a750 | 98362 | WA |
| 70584e4b-1a71-432c-ad20-c2419674e592 | Puyallup | Pierce | US | 11515 Canyon Road East | null | null | 37dbdc2d-e579-4efa-984b-943847fce24d | 98373 | WA |

## ORGANIZATION TABLE

| id | url | name | email | tax_id | tax_status | description | legal_status | alternate_name |
|----|-----|------|-------|--------|------------|-------------|--------------|----------------|
| 51d87478-eb86-4ab3-871c-ec1a0287675a | https://trinitytransitionalhousing.com/ | Trinity Transitional Housing | null | null | null | Provides a safe and supportive environment for individuals in recovery from substance abuse. Our mission is to help our residents achieve a successful transition back into society. | null | null |
| e64adfc7-0535-40de-bac8-47b80c37466d | https://www.casapartners.org/ | CASA Partners | null | null | null | Dedicated to supporting the local Court Appointed Special Advocate (C.A.S.A.) Program and enhancing the lives of abused and neglected children, as well as at-risk youth, in the Spokane County area. | null | null |
| 64fbe141-a840-4938-a9d1-b577a20c55dd | https://www.immaculateconceptiondavenport.com/ | Immaculate Conception Catholic Parish | null | null | null | Provides a hot lunch on Thursdays to anyone in the community. First come, first serve. | null | null |
| 68ce4e46-5d2d-4b25-b0b3-09a5d7f94e1f | https://www.capeco-works.org/ | Community Action Program of East Central Oregon | null | null | null | Provides multiple services across the following sectors: food and nutrition, housing solutions, services for independence, aging, and youth. | null | null |
| 7f04111c-8952-4d64-b9b4-8e3ba6919cc8 | https://nationalparentyouthhelpline.org/ | National Parent and Youth Helpline | null | null | null | Provides emotional support, problem solving goals and tools, and encouragement for parents, youth, and caregivers. | null | null |

## LOCATION TABLE

| id | name | latitude | longitude | description | location_type | alternate_name | organization_id |
|----|------|----------|-----------|-------------|---------------|----------------|----------------|
| 5098ccf5-2381-467f-9307-9589f076ade7 | zzz - Cascade Community Healthcare - Phoenix Clubhouse | 46.716863 | -122.95602 | null | physical | null | f97c1b2d-21f2-425e-8213-48166e8d1b3c |
| 2cfea6d0-4737-488f-b0b3-5109ee1430b2 | zzz - Vashon Presbyterian Church | 47.446054 | -122.46013 | null | physical | null | 7018e026-0869-45c9-8a21-ad2a9b0942d1 |
| 4524afe6-1f10-4325-bfa4-0aa135ff15c1 | Washington State Department of Children, Youth, and Families in North Spokane | 47.720115 | -117.394587 | null | physical | null | 5342d497-abdd-4fb3-895c-0c0145f76e21 |
| a09c55ba-38a6-4fcd-bba2-7d5606c8cb07 | *Main Site | 47.57176 | -122.631868 | null | physical | null | 0c5bbb95-c3ed-48b7-811a-50edb6ed4794 |
| 47660b38-0816-4598-ab2c-e86a366d63c7 | Community Events - DSHS Offices | 47.240997 | -122.464035 | null | physical | null | c8dad105-1d78-45da-b040-1cccbb1c0ddc |

## SERVICE TABLE

| id | url | name | email | status | description | alternate_name | organization_id |
|----|-----|------|-------|--------|-------------|----------------|----------------|
| 9769064c-b871-4fdc-859e-d80a3fe94e0f | https://nhwa.org/aging-and-disability-services/ | Family Caregiver Support Program offered by Neighborhood House at Wiley Center at Greenbridge | romanf@nhwa.org | active | Offer supportive services to unpaid adult caregivers. Caregiver support services may include:<br>-  Individualized comprehensive needs assessments and develop care plans<br>-  In-home or office visits within King County<br>-  Referrals to support groups, counseling and other resources<br>-  Advice on the use of supplies and equipment<br>-  Limited respite for the caregiver | Neighborhood House, CAP, KCCSN, Neighborhood House - High Point, Neighborhood House - High Point Early Childhood Center, Neighborhood House - Lee House at NewHolly, Neighborhood House - NewHolly Early Childhood Center, Neighborhood House - Rainier Vista, Neighborhood House - Wiley Center at Greenbridge, Neighborhood House - Yesler Terrace, NH, Family Caregiver Support Program, Neighborhood House - Central Office, Sunset Neighborhood Center | 4f5628ca-c7e5-4cc3-aac9-51b8701f7e21 |
| b162e9d9-3da1-42fe-9816-8e6fef242cce | https://nhwa.org/aging-and-disability-services/ | Family Caregiver Support Program offered by Neighborhood House at Birch Creek Career Center | romanf@nhwa.org | active | Offer supportive services to unpaid adult caregivers. Caregiver support services may include:<br>-  Individualized comprehensive needs assessments and develop care plans<br>-  In-home or office visits within King County<br>-  Referrals to support groups, counseling and other resources<br>-  Advice on the use of supplies and equipment<br>-  Limited respite for the caregiver | Neighborhood House, CAP, KCCSN, Neighborhood House - High Point, Neighborhood House - High Point Early Childhood Center, Neighborhood House - Lee House at NewHolly, Neighborhood House - NewHolly Early Childhood Center, Neighborhood House - Rainier Vista, Neighborhood House - Wiley Center at Greenbridge, Neighborhood House - Yesler Terrace, NH, Family Caregiver Support Program, Neighborhood House - Central Office, Sunset Neighborhood Center | 4f5628ca-c7e5-4cc3-aac9-51b8701f7e21 |
| 4113fbab-51ad-423d-a702-9c1cff74e326 | https://arcofkingcounty.org/resource-guide/immigration-status-benefits/immigration-status-benefits.html | Information & Reading Materials on Immigration offered at Arc of King County | null | active | Maintains a web page featuring materials regarding immigration, legal information, and eligibility for benefits,  as well as legal resources and community supports. | Arc of King County, KCPFC, P2P, Parent 2 Parent, Parent to Parent of King County, The Arc of King County, Resource Center, Arc of King County | d8c3508a-cc7b-46ad-b174-fb063004bf72 |
| 2c9a3ebb-4421-437d-a9d3-04d7300433d9 | https://crpolice.org/ | Animal Control offered at City of Castle Rock | null | active | Provides help to resolve animal control problems such as stray or vicious animals, animal-related emergencies, and reports of animal cruelty. | Animal Control, City of Castle Rock | b664dea5-2638-4828-b686-8eaca246560d |
| ed7b5330-a6b5-455d-95b8-e60a578dc88f | https://www.ci.woodland.wa.us/police | Animal Control offered by City of Woodland at Police Department | null | active | Provides help to resolve animal control problems such as stray or vicious animals, animal-related emergencies, and reports of animal cruelty. | Animal Control, City of Woodland, Police Department | 8006d4b1-46ff-4e41-bfe1-99762f13caff |

## PHONE TABLE

| id | type | number | language | extension | service_id | location_id | organization_id | service_at_location_id |
|----|------|--------|----------|-----------|------------|-------------|----------------|------------------------|
| 7761c466-c877-4262-a800-f1761287fe3c | voice | (253) 833-7444 | en | null | null | null | null | e6238950-0952-41f6-85ba-5249984c298e |
| 8beabdec-705b-4f20-a0be-5887a22da673 | voice | (253) 833-7444 | en | null | null | null | null | a2b57e4f-4c87-4cf9-8231-7ce1f8d7a6c8 |
| c81ad77a-11c2-497c-ab4d-fb94847643eb | voice | (253) 833-7444 | en | null | null | null | null | 2552a76f-5bd5-47b7-bdfd-ec94a0a22614 |
| 84ff7a7d-f81b-4570-ba61-fa11f373466d | voice | (253) 833-7444 | en | null | null | null | null | aab41ca0-bffc-46c4-8171-9e4219360a08 |
| 52820f3a-72c8-42e8-bc27-6c0a561d1996 | voice | (253) 833-7444 | en | null | null | null | null | ac76bf6e-b4f4-45cc-89ec-ef4d9227ee69 |

## SERVICE_AT_LOCATION TABLE

| id | service_id | description | location_id |
|----|------------|-------------|-------------|
| bc881536-7063-432a-962e-fc43840f761f | 6ec3a6f0-1e23-4fcc-a06e-c53a2e52ad2a | null | 2f6f1bad-c5d1-4729-9773-525a7277a489 |
| 0003d7d7-c6f4-482c-8b35-a8bac6e23b1b | 0f5a9013-239b-4c73-ab05-61778c1bfee2 | null | 9220422d-e566-4524-8d44-cb021e4e38de |
| b20ef1c4-6af7-4716-b551-57b0b9682814 | b55c1cc1-7e3a-4fc9-9be1-95f727528d2e | null | c9e37c2a-4c19-47c9-aaa6-9302cc504e56 |
| b83ceba6-2d27-43d7-9909-fd1236f51f14 | e1d94d4b-8b3c-445d-86f7-e0d98df4d0a1 | null | bb2495a9-a64c-4638-95f8-8d8ea1a8b21e |
| 6016b4a0-eb44-429d-9553-84d41e726834 | b16ec317-8907-4cd1-9934-93cf36250dd5 | null | 662230b9-659d-4be3-acf6-899cbf15ff98 |

## MATCH_CLUSTERS TABLE

| id | notes | reasoning | confidence | is_reviewed | reviewed_at | reviewed_by | review_result |
|----|-------|-----------|------------|-------------|-------------|-------------|---------------|
| 927ac07b-c152-4144-9f11-dfbe194f3732 | Matched on normalized email: president@thetacomaurbanleague.org; Matched on normalized domain: thetacomaurbanleague.org | Deterministic matching only (Phase 1) | 1 | false | null | null | null |
| 9d8ff2c2-fe68-40ae-8a4d-4ed350b9ea8d | Matched on normalized email: sheilag@spokaneneyc.org; Matched on normalized domain: spokaneneyc.com | Deterministic matching only (Phase 1) | 1 | false | null | null | null |
| c7a2fa02-4c28-43e3-baf2-fc891af8f85d | Matched on normalized email: info@ilca.org | Deterministic matching only (Phase 1) | 1 | false | null | null | null |
| 34e18cc9-4aea-4faa-bfe6-6f96d3144d69 | Matched on normalized email: office@firstnaz.com; Matched on normalized phone: 2087439501; Matched on normalized domain: firstnaz.com | Deterministic matching only (Phase 1) | 1 | false | null | null | null |
| eb53241f-da5b-4e06-a89d-970682353fa0 | Matched on normalized email: info@forahealth.org; Matched on normalized domain: forahealth.org | Deterministic matching only (Phase 1) | 1 | false | null | null | null |

## CLUSTER_ENTITIES TABLE

| id | entity_id | cluster_id | created_at | entity_type |
|----|-----------|------------|------------|-------------|
| 1a0b8180-1d05-4c0d-afb1-d90c4266ced3 | 1e013a81-5ddc-4377-9bfd-0d89643d9054 | c7a2fa02-4c28-43e3-baf2-fc891af8f85d | 2025-04-08T19:33:00.82747+00:00 | service |
| 7b5796ed-c9ce-4810-b4c2-94acd9f0683d | 6456c333-5cc6-4cad-8d11-94d1022a1d9f | c7a2fa02-4c28-43e3-baf2-fc891af8f85d | 2025-04-08T19:33:00.82747+00:00 | organization |
| 02c8de04-b5ea-451f-88a3-2b8459625316 | 0779b9d6-3f61-4f1d-a9ad-3d1752213d0c | 5615ccfc-2709-4ded-a991-a118e9128f6c | 2025-04-08T19:33:00.82747+00:00 | service |
| 491fde4f-8f2f-4c90-a8a6-6b30083d2322 | dc32db64-a8b3-44ce-9ba0-a0c472804f4e | 5615ccfc-2709-4ded-a991-a118e9128f6c | 2025-04-08T19:33:00.82747+00:00 | organization |
| 3362011c-4cc5-4969-b4a0-a3ffd5f9cd9f | 781b6972-3d0e-4ae2-82cc-f6e985ce5f1d | 5615ccfc-2709-4ded-a991-a118e9128f6c | 2025-04-08T19:33:00.82747+00:00 | service |

## MATCHING_METHODS TABLE

| id | details | cluster_id | confidence | method_name |
|----|---------|------------|------------|-------------|
| 4348d57c-d4c3-4555-bc2d-5e087af5f7ff | null | 45571ab0-c20b-4877-8f8f-ff3f16a838aa | 1 | Phone |
| 1e1a1925-fdfa-41b8-b04b-3b0c4c4f2dc5 | null | 45571ab0-c20b-4877-8f8f-ff3f16a838aa | 1 | Email |
| ce214275-42a0-4600-9a20-c4c2d559d27b | null | 45571ab0-c20b-4877-8f8f-ff3f16a838aa | 1 | URL |
| 079b085b-010a-4c70-8ada-614d36a4b806 | null | 31c279af-45a4-4e9a-be79-99f236785e45 | 1 | Phone |
| 7e725552-98bf-45b8-8c5c-ea34c4fae4b9 | null | 31c279af-45a4-4e9a-be79-99f236785e45 | 1 | URL |