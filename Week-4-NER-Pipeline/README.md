# Week 4: Designing an NLP Pipeline for Named Entity Recognition (NER)

## Project: CivicSift AI

### Objective
The objective of Week 4 is to design an end-to-end Named Entity Recognition (NER) pipeline that extracts and categorizes important information from citizen complaints.

### Work Completed
- Designed the complete NER pipeline architecture.
- Defined standard entities: PERSON, LOCATION, ORGANIZATION, DATE, and TIME.
- Defined custom civic entities: ISSUE and INFRASTRUCTURE.
- Explained multilingual text preprocessing techniques.
- Reviewed suitable multilingual NER models, including mBERT, XLM-RoBERTa, and IndicBERT.
- Documented post-processing, entity correction, and validation strategies.
- Discussed contextual ambiguity, polysemy, error handling, and accuracy improvement.
- Prepared a pipeline flowchart and implementation roadmap.

### Dataset Used
**NYC 311 Complaints Dataset**

- Total records: 369
- Columns: `text`, `complaint_type`
- Used as a source of civic complaint vocabulary and issue categories.

**Note:** The supplied dataset does not contain token-level NER annotations. Additional annotated data is required for supervised NER model training and evaluation.

### Proposed Pipeline
1. Raw Complaint Input
2. Language Identification
3. Text Preprocessing
4. NER Model
5. Post-processing
6. Validation
7. Structured Output

### Deliverables
- Detailed NER Pipeline Design Report (DOCX)
- NER Pipeline Flowchart
- Implementation Roadmap

### Status
**Week 4: Completed — NER Pipeline Design and Documentation**

This week's work focuses on pipeline design and implementation strategy. Model training and deployment are not claimed as completed.
