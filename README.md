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

or just open the project with visual studios.


to run this program
## 1. Build everything
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
cmake --build . --config Release

## 2. Put some images in data/raw/ (at least a few hundred)

## 3. Preprocess once
./preprocess

## 4. Train (will auto-resume from checkpoint if one exists)
./train

## 5. Generate from a checkpoint
./generate checkpoints/latest.bin data/raw/some_image.jpg output.png
