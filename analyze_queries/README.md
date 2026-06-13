# Query Processing System

This system processes queries from the "Workflow Evaluation_Table_Grid.xlsx" file using OpenRouter API (Gemini Flash) to provide:

1. **Chinese Translation** of each query
2. **Difficulty Level** assessment (1-5 scale)
3. **Executability** evaluation with reasoning

## 🚀 Quick Start

1. **Run the demo** (no API key needed):
   ```bash
   python demo_without_api.py
   ```

2. **Set up API key** and test:
   ```bash
   $env:OPENROUTER_API_KEY="your_api_key_here"
   python test_sample.py
   ```

3. **Process all queries**:
   ```bash
   python process_queries.py
   ```

## 📁 Files Description

| File | Purpose |
|------|---------|
| `process_queries.py` | Main processing script |
| `test_sample.py` | Test with first 3 rows |
| `demo_without_api.py` | Demo structure without API calls |
| `check_columns.py` | Examine Excel file structure |
| `setup_guide.md` | Detailed setup instructions |

## 📊 Input/Output

**Input:** `Workflow Evaluation_Table_Grid.xlsx` (56 rows, 20 columns)

**Output:** `Workflow_Evaluation_Processed.csv` with 5 additional columns:
- `Chinese_Translation`: Query translated to Chinese
- `Difficulty_Level`: 1-5 difficulty rating
- `Difficulty_Reasoning`: Explanation of difficulty assessment
- `Is_Executable`: True/False executability
- `Executability_Reason`: Detailed explanation

## 🎯 Difficulty Scale

- **1**: Very easy - Simple information retrieval
- **2**: Easy - Straightforward tasks with basic reasoning
- **3**: Medium - Moderate reasoning, multi-step processes
- **4**: Hard - Complex tasks requiring advanced reasoning
- **5**: Very hard - Tasks pushing LLM capability limits

## 🔧 Features

- **Robust Error Handling**: Retry logic with exponential backoff
- **Progress Tracking**: Auto-save every 10 rows
- **Rate Limiting**: 1-second delay between API calls
- **JSON Response Parsing**: Structured data extraction
- **UTF-8 Support**: Proper Chinese character encoding

## 📈 Sample Output

```csv
Diligent Query,Chinese_Translation,Difficulty_Level,Difficulty_Reasoning,Is_Executable,Executability_Reason
"I am the head of influencer marketing...","我是Wegic的网红营销主管...",3,"Requires multi-step planning and analysis",true,"Clear objectives and context provided"
```

## 🛠️ Technical Details

- **API**: OpenRouter with Gemini Flash 1.5
- **Rate Limit**: 1 request/second
- **Timeout**: 60 seconds per request
- **Encoding**: UTF-8 with BOM for Excel compatibility
- **Progress**: Automatic checkpointing

## ⚠️ Prerequisites

```bash
pip install pandas openpyxl requests
```

Get API key from: https://openrouter.ai/

## 📞 Support

For issues or questions, check the error messages and progress files created during processing. 