# imagegenCUDA

Open the project in the Visual Studio x64 command prompt, then run `code .` to open it with the CUDA compiler configuration.

Create `.vscode/tasks.json` with this content:

```json
{
    "version": "2.0.0",
    "tasks": [
        {
            "label": "Build CUDA",
            "type": "process",
            "command": "nvcc",
            "args": [
                "-g",
                "-G",
                "-ccbin",
                "C:/Program Files/Microsoft Visual Studio/2022/Community/VC/Tools/MSVC/14.xx.xxxx/bin/Hostx64/x64",
                "${file}",
                "-o",
                "${fileDirname}/${fileBasenameNoExtension}.exe"
            ],
            "group": {
                "kind": "build",
                "isDefault": true
            },
            "problemMatcher": [
                "$nvcc"
            ]
        }
    ]
}
```

After that, you can build and run `.cu` or CUDA  files from VS Code.
