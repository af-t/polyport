use serde::{Deserialize, Serialize};
use serde_json::Value;

pub const GEMINI_INLINE_BUDGET_BYTES: usize = 7 * 1024 * 1024;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum MediaKind {
    Image,
    Document,
    Audio,
    Video,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MediaPart {
    pub kind: MediaKind,
    pub mime_type: String,
    pub base64: Option<String>,
    pub url: Option<String>,
    pub file_id: Option<String>,
    pub filename: Option<String>,
}

#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub enum MediaError {
    #[error("{0} input is not supported by provider")]
    UnsupportedMedia(String),
    #[error("media input is {bytes} bytes, exceeding the inline limit of {limit} bytes")]
    MediaTooLarge { bytes: usize, limit: usize },
    #[error("could not fetch media at {0}")]
    MediaFetchFailed(String),
}

fn parse_data_uri(s: &str) -> Option<(&str, &str)> {
    if !s.starts_with("data:") {
        return None;
    }
    let rest = &s[5..];
    let (mime_part, data_part) = rest.split_once(";base64,")?;
    Some((mime_part, data_part))
}

fn default_mime_for_kind(kind: MediaKind) -> &'static str {
    match kind {
        MediaKind::Image => "image/*",
        MediaKind::Audio => "audio/*",
        MediaKind::Video => "video/*",
        MediaKind::Document => "application/pdf",
    }
}

fn media_from_url_or_data(kind: MediaKind, url: Option<&str>) -> Option<MediaPart> {
    let url = url?;
    if let Some((mime, data)) = parse_data_uri(url) {
        return Some(MediaPart {
            kind,
            mime_type: mime.to_string(),
            base64: Some(data.to_string()),
            url: None,
            file_id: None,
            filename: None,
        });
    }

    Some(MediaPart {
        kind,
        mime_type: default_mime_for_kind(kind).to_string(),
        base64: None,
        url: Some(url.to_string()),
        file_id: None,
        filename: None,
    })
}

#[must_use]
pub fn parse_openai_media_block(block: &Value) -> Option<MediaPart> {
    let obj = block.as_object()?;
    let block_type = obj.get("type")?.as_str()?;

    match block_type {
        "image_url" => {
            let img = obj.get("image_url")?;
            let url_str = img.as_str().or_else(|| img.get("url").and_then(Value::as_str));
            media_from_url_or_data(MediaKind::Image, url_str)
        }
        "video_url" => {
            let vid = obj.get("video_url")?;
            let url_str = vid.as_str().or_else(|| vid.get("url").and_then(Value::as_str));
            media_from_url_or_data(MediaKind::Video, url_str)
        }
        "input_audio" => {
            let audio = obj.get("input_audio")?;
            let data = audio.get("data")?.as_str()?;
            let format_str = audio.get("format").and_then(Value::as_str).unwrap_or("wav");
            let mime = if format_str == "mp3" { "audio/mpeg" } else { "audio/wav" };
            Some(MediaPart {
                kind: MediaKind::Audio,
                mime_type: mime.to_string(),
                base64: Some(data.to_string()),
                url: None,
                file_id: None,
                filename: None,
            })
        }
        "file" => {
            let file = obj.get("file")?;
            let file_id = file.get("file_id").and_then(Value::as_str);
            let filename = file.get("filename").and_then(Value::as_str).map(String::from);

            if let Some(fid) = file_id {
                return Some(MediaPart {
                    kind: MediaKind::Document,
                    mime_type: "application/pdf".to_string(),
                    base64: None,
                    url: None,
                    file_id: Some(fid.to_string()),
                    filename,
                });
            }

            let src = file.get("file_data").and_then(Value::as_str)?;
            if let Some((mime, data)) = parse_data_uri(src) {
                Some(MediaPart {
                    kind: MediaKind::Document,
                    mime_type: mime.to_string(),
                    base64: Some(data.to_string()),
                    url: None,
                    file_id: None,
                    filename,
                })
            } else {
                Some(MediaPart {
                    kind: MediaKind::Document,
                    mime_type: "application/pdf".to_string(),
                    base64: None,
                    url: Some(src.to_string()),
                    file_id: None,
                    filename,
                })
            }
        }
        _ => None,
    }
}

#[must_use]
pub fn parse_anthropic_media_block(block: &Value) -> Option<MediaPart> {
    let obj = block.as_object()?;
    let block_type = obj.get("type")?.as_str()?;

    match block_type {
        "image" => {
            let src = obj.get("source")?;
            let src_type = src.get("type")?.as_str()?;
            if src_type == "base64" {
                let media_type = src.get("media_type")?.as_str()?;
                let data = src.get("data")?.as_str()?;
                Some(MediaPart {
                    kind: MediaKind::Image,
                    mime_type: media_type.to_string(),
                    base64: Some(data.to_string()),
                    url: None,
                    file_id: None,
                    filename: None,
                })
            } else if src_type == "url" {
                let url = src.get("url")?.as_str()?;
                let media_type = src.get("media_type").and_then(Value::as_str).unwrap_or("image/*");
                Some(MediaPart {
                    kind: MediaKind::Image,
                    mime_type: media_type.to_string(),
                    base64: None,
                    url: Some(url.to_string()),
                    file_id: None,
                    filename: None,
                })
            } else {
                None
            }
        }
        "document" => {
            let src = obj.get("source")?;
            let src_type = src.get("type")?.as_str()?;
            let mime = src
                .get("media_type")
                .and_then(Value::as_str)
                .unwrap_or("application/pdf");
            let filename = obj.get("title").and_then(Value::as_str).map(String::from);

            match src_type {
                "base64" => {
                    let data = src.get("data")?.as_str()?;
                    Some(MediaPart {
                        kind: MediaKind::Document,
                        mime_type: mime.to_string(),
                        base64: Some(data.to_string()),
                        url: None,
                        file_id: None,
                        filename,
                    })
                }
                "url" => {
                    let url = src.get("url")?.as_str()?;
                    Some(MediaPart {
                        kind: MediaKind::Document,
                        mime_type: mime.to_string(),
                        base64: None,
                        url: Some(url.to_string()),
                        file_id: None,
                        filename,
                    })
                }
                "file" => {
                    let file_id = src.get("file_id")?.as_str()?;
                    Some(MediaPart {
                        kind: MediaKind::Document,
                        mime_type: mime.to_string(),
                        base64: None,
                        url: None,
                        file_id: Some(file_id.to_string()),
                        filename,
                    })
                }
                _ => None,
            }
        }
        _ => None,
    }
}

/// Renders a media part into Gemini inlineData part format.
///
/// # Errors
/// Returns `MediaError` if media exceeds the 7MB inline limit or cannot be resolved inline.
pub fn render_for_gemini(part: &MediaPart) -> Result<Value, MediaError> {
    if let Some(b64) = &part.base64 {
        let bytes = (b64.len() * 3) / 4;
        if bytes > GEMINI_INLINE_BUDGET_BYTES {
            return Err(MediaError::MediaTooLarge {
                bytes,
                limit: GEMINI_INLINE_BUDGET_BYTES,
            });
        }
        return Ok(serde_json::json!({
            "inlineData": {
                "mimeType": &part.mime_type,
                "data": b64
            }
        }));
    }

    if let Some(url) = &part.url {
        if let Some((mime, b64)) = parse_data_uri(url) {
            let bytes = (b64.len() * 3) / 4;
            if bytes > GEMINI_INLINE_BUDGET_BYTES {
                return Err(MediaError::MediaTooLarge {
                    bytes,
                    limit: GEMINI_INLINE_BUDGET_BYTES,
                });
            }
            return Ok(serde_json::json!({
                "inlineData": {
                    "mimeType": mime,
                    "data": b64
                }
            }));
        }
        return Err(MediaError::MediaFetchFailed(url.clone()));
    }

    Err(MediaError::UnsupportedMedia(format!("{:?}", part.kind)))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_openai_image_url_base64_and_url() {
        let b64_block = serde_json::json!({
            "type": "image_url",
            "image_url": {
                "url": "data:image/png;base64,iVBORw0KGgoAAAANSUhEUg=="
            }
        });
        let part = parse_openai_media_block(&b64_block).unwrap();
        assert_eq!(part.kind, MediaKind::Image);
        assert_eq!(part.mime_type, "image/png");
        assert_eq!(part.base64.as_deref(), Some("iVBORw0KGgoAAAANSUhEUg=="));

        let url_block = serde_json::json!({
            "type": "image_url",
            "image_url": "https://example.com/photo.jpg"
        });
        let part2 = parse_openai_media_block(&url_block).unwrap();
        assert_eq!(part2.kind, MediaKind::Image);
        assert_eq!(part2.url.as_deref(), Some("https://example.com/photo.jpg"));
    }

    #[test]
    fn test_parse_anthropic_image_base64() {
        let block = serde_json::json!({
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": "image/jpeg",
                "data": "jpegbase64content"
            }
        });
        let part = parse_anthropic_media_block(&block).unwrap();
        assert_eq!(part.kind, MediaKind::Image);
        assert_eq!(part.mime_type, "image/jpeg");
        assert_eq!(part.base64.as_deref(), Some("jpegbase64content"));
    }

    #[test]
    fn test_render_for_gemini_inline_data() {
        let part = MediaPart {
            kind: MediaKind::Image,
            mime_type: "image/png".to_string(),
            base64: Some("pngbase64".to_string()),
            url: None,
            file_id: None,
            filename: None,
        };
        let gemini_part = render_for_gemini(&part).unwrap();
        assert_eq!(
            gemini_part,
            serde_json::json!({
                "inlineData": {
                    "mimeType": "image/png",
                    "data": "pngbase64"
                }
            })
        );
    }
}
