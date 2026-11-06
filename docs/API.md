# API Reference & Documentation

## Endpoints

### Health & Diagnostics
- `GET /api/v1/health`
  - Returns service status, hardware profile, active device, and memory usage.

### Video Ingestion & Jobs
- `POST /api/v1/uploads`
  - Upload video file (MP4, MOV, AVI, WebM). Validates header, file size, codec, and writes to `data/uploads/`.
- `POST /api/v1/jobs`
  - Submit reconstruction job:
    ```json
    {
      "video_id": "upload_abc123",
      "morphology": "human",
      "target_fps": 15,
      "export_formats": ["gltf", "bvh", "blender"]
    }
    ```
- `GET /api/v1/jobs/{job_id}`
  - Poll job progress, current stage, and status (`QUEUED`, `PROCESSING`, `COMPLETED`, `FAILED`).
- `POST /api/v1/jobs/{job_id}/cancel`
  - Cancel an active job.

### Assets & Reconstruction
- `GET /api/v1/rigs/{rig_id}`
  - Retrieve reconstructed skeletal topology, joint names, bone hierarchy, and rest pose.
- `GET /api/v1/animations/{animation_id}`
  - Retrieve animation keyframe data (bone quaternions, root translation).
- `GET /api/v1/exports/{export_id}/download`
  - Download generated `.gltf`, `.glb`, `.bvh`, or Blender `.py` script.
