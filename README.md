
# Linear Regression MCP

Welcome to **Linear Regression MCP**! This project demonstrates an end-to-end machine learning workflow using Claude and the Model Context Protocol (MCP). 

**Claude** can train a **Linear Regression model** entirely by itself, simply by uploading a CSV file containing the dataset. The system goes through the entire **ML model training lifecycle**, handling data preprocessing, training, and evaluation (RMSE calculation).


<br>

## Setup and Installation

### 1. Clone the Repository:

First, clone the repository to your local machine:

```bash
git clone https://github.com/Tirth-Koradiya/Linear-Regression-MCP.git
cd Linear-Regression-MCP
```

### 2. Install `uv`:

`uv` is an extremely fast Python package and project manager, written in Rust. It is essential for managing the server and dependencies in this project.

- Download and install `uv` from [here](https://docs.astral.sh/uv/#installation).

### 3. Install Dependencies:

Once uv is installed, run the following command to install all necessary dependencies:

```bash
uv sync
```

### 4. Configure Claude Desktop:

To integrate the server with Claude Desktop, you will need to modify the Claude configuration file. Follow the instructions for your operating system:

- For macOS or Linux:

```bash
code ~/Library/Application\ Support/Claude/claude_desktop_config.json
```

- For Windows:

```bash
code $env:AppData\Claude\claude_desktop_config.json
```

- In the configuration file, locate the `mcpServers` section, and replace the placeholder paths with the absolute paths to your `uv` installation and the Linear Regression project directory. It should look like this:

```bash
{
    "mcpServers":
    {
        "linear-regression": 
        {
            "command": "ABSOLUTE/PATH/TO/.local/bin/uv",
            "args":
            [
                "--directory",
                "ABSOLUTE/PATH/TO/YOUR-LINEAR-REGRESSION-REPO", 
                "run",
                "server.py"
            ] 
        }
    }
}
```

- Once the file is saved, restart Claude Desktop to link with the MCP server.

<br>

## Available Tools

The following tools are available in this project to help you work with the dataset and train the model:

| **Tool**                                      | **Description**                                                                 | **Arguments**                                       |
|-----------------------------------------------|---------------------------------------------------------------------------------|----------------------------------------------------|
| `upload_file(path)`                           | Uploads a CSV file and stores it for processing.                                | `path`: Absolute path to the CSV file.             |
| `get_columns_info()`                          | Retrieves the column names in the uploaded dataset.                             | No arguments.                                      |
| `check_category_columns()`                    | Checks for any categorical columns in the dataset.                              | No arguments.                                      |
| `label_encode_categorical_columns()`          | Label encodes categorical columns into numerical values.                        | No arguments.                                      |
| `train_linear_regression_model(output_column)`| Trains a linear regression model and calculates RMSE.                          | `output_column`: The name of the target column.    |



