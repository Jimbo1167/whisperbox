# Future Enhancements

## Core Features

### Transcription Improvements
- [x] Add support for batch processing multiple videos (`scripts/transcribe.py batch`)
- [ ] Implement real-time transcription for live video streams
- [ ] Add support for custom Whisper model fine-tuning
- [ ] Implement automatic language detection (Parakeet detects the language itself; Whisper still uses `LANGUAGE`)
- [ ] Add support for multi-language transcription in the same video
- [ ] Implement confidence scores for transcriptions

### Speaker Diarization Enhancements
- [ ] Add speaker identification (assign names to speakers)
- [ ] Implement speaker embedding for consistent speaker labels across multiple videos
- [ ] Add support for pre-registered speaker voices
- [ ] Improve speaker segmentation for overlapping speech
- [ ] Add gender detection for speakers

### Performance Optimizations
- [ ] Implement parallel processing for audio extraction
- [ ] Add GPU memory optimization for longer videos
- [x] Implement streaming transcription to handle large files (`stream` command)
- [ ] Add support for distributed processing
- [ ] Optimize memory usage during diarization

### Output Formats
- [ ] Add JSON output format with confidence scores
- [ ] Implement automatic subtitle generation for YouTube
- [ ] Add support for custom timestamp formats
- [x] Implement word-level timestamps (`stream --words`)
- [ ] Add support for EDL (Edit Decision List) format
- [ ] Add support for TTML (Timed Text Markup Language)

## User Experience

### CLI Improvements
- [ ] Add progress bars for all processing steps (`stream`, `batch` and the model client have them; `transcribe` prints none in its default `pretty` mode, and `--progress jsonl` emits machine-readable events)
- [x] Unified CLI with subcommands (`scripts/transcribe.py`)
- [ ] Implement command-line arguments for all options (beam size, CPU threads, cache, `FORCE_CPU`, timeouts and others are still `.env`-only)
- [ ] Add interactive mode for configuration
- [ ] Implement resume capability for interrupted processes
- [ ] Add preview mode for quick sample transcription

### GUI Features
- [x] Browser UI for drag-and-drop uploads (served by the model server)
- [ ] Create a desktop application interface
- [ ] Add real-time visualization of transcription
- [ ] Implement waveform display with transcription
- [ ] Add interactive speaker label editing
- [ ] Implement transcript editor with time alignment

### API Integration
- [x] Create REST API for remote transcription (`scripts/model_server.py`)
- [ ] Add WebSocket support for real-time updates
- [ ] Implement cloud storage integration (S3, GCS)
- [ ] Add support for video platform APIs (YouTube, Vimeo)
- [ ] Create webhook notifications for completion

## Quality Improvements

### Audio Processing
- [ ] Add noise reduction preprocessing
- [ ] Implement automatic gain control
- [ ] Add support for different audio codecs
- [ ] Implement audio enhancement filters
- [ ] Add voice activity detection optimization

### Accuracy Enhancements
- [ ] Add custom vocabulary support
- [ ] Implement context-aware corrections
- [ ] Add support for industry-specific terminology
- [ ] Implement automatic punctuation correction
- [ ] Add support for speaker-specific language models

## Development Tools

### Testing
- [ ] Add integration tests that run real models on real audio (the suite in `tests/unit` uses fake models; `tests/fixtures` holds generated media)
- [ ] Implement performance benchmarking suite
- [x] Add automated accuracy testing (`scripts/benchmark.py`, see benchmarks/README.md)
- [ ] Create test data generation tools
- [ ] Implement CI/CD pipeline

### Documentation
- [x] Add API documentation (docs/api/)
- [x] Create user guides for different use cases (docs/user_guide/)
- [ ] Add performance tuning guide
- [x] Create troubleshooting guide (docs/user_guide/index.md#troubleshooting)
- [ ] Add architecture documentation

## Deployment

### Containerization
- [x] Create Docker container
- [ ] Add Kubernetes deployment configurations
- [ ] Implement container optimization for GPU support
- [ ] Add multi-stage build process
- [ ] Create deployment automation scripts

### Monitoring
- [ ] Add system resource monitoring (the CLI prints an average CPU/memory summary after each run)
- [ ] Implement error tracking and reporting
- [ ] Add performance metrics collection
- [ ] Create dashboard for system status
- [ ] Implement automated alerts

## Research Areas

### AI/ML Improvements
- [x] Investigate newer speech recognition models (Parakeet engine added)
- [ ] Research improved diarization techniques
- [ ] Explore emotion detection in speech
- [ ] Study accent recognition and adaptation
- [ ] Research multilingual model improvements

### Experimental Features
- [ ] Explore real-time translation
- [ ] Investigate speaker age estimation
- [ ] Research acoustic scene analysis
- [ ] Explore background noise classification
- [ ] Investigate speech enhancement techniques 