package main

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"net/http"
	"net/url"
	"os"
	"strconv"
	"strings"
	"time"

	"go.opentelemetry.io/contrib/instrumentation/net/http/otelhttp"
)

var BuildVersion = "dev"

func getEnvWithDefault(key, defaultValue string) string {
	value := os.Getenv(key)
	if value == "" {
		return defaultValue
	}
	return value
}

func getEnvInt(key string, defaultValue int) int {
	value := os.Getenv(key)
	if value == "" {
		return defaultValue
	}
	parsed, err := strconv.Atoi(value)
	if err != nil {
		return defaultValue
	}
	return parsed
}

func getEnvFloat(key string, defaultValue float64) float64 {
	value := os.Getenv(key)
	if value == "" {
		return defaultValue
	}
	parsed, err := strconv.ParseFloat(value, 64)
	if err != nil {
		return defaultValue
	}
	return parsed
}

// pixmeClient wraps http.Client with the internal API key header for pixme-api requests.
type pixmeClient struct {
	client *http.Client
	apiKey string
}

func (c *pixmeClient) Get(url string) (*http.Response, error) {
	req, err := http.NewRequest(http.MethodGet, url, nil)
	if err != nil {
		return nil, err
	}
	if c.apiKey != "" {
		req.Header.Set("X-Internal-Key", c.apiKey)
	}
	return c.client.Do(req)
}

func (c *pixmeClient) Post(url, contentType string, body io.Reader) (*http.Response, error) {
	req, err := http.NewRequest(http.MethodPost, url, body)
	if err != nil {
		return nil, err
	}
	if contentType != "" {
		req.Header.Set("Content-Type", contentType)
	}
	if c.apiKey != "" {
		req.Header.Set("X-Internal-Key", c.apiKey)
	}
	return c.client.Do(req)
}

type ImageApiResponse struct {
	Images ImageList `json:"images"`
	Count  int       `json:"count"`
	Limit  int       `json:"limit"`
	Offset int       `json:"offset"`
	Total  int       `json:"total"`
}

type ImageList []ImageDescriptor
type CategoryList []string

type ImageDescriptor struct {
	ID            string       `json:"id"`
	Name          string       `json:"name"`      // This is the name of the image file
	Thumbnail     string       `json:"thumbnail"` // This is the path to the thumbnail image
	ThumbnailURI  string       `json:"thumbnailUri"`
	ThumbnailPath string       `json:"-"` // This is the path to the thumbnail image file
	Title         string       `json:"title"`
	Description   string       `json:"description"`
	URI           string       `json:"uri"`        // This is the uri to the full image
	FilePath      string       `json:"-"`          // This is the path to the full image file
	Categories    CategoryList `json:"categories"` // These are the categories associated with the image
}

type DeepFaceRequest struct {
	Img              string `json:"img"`
	DbPath           string `json:"db_path"` // This is the path to the image in the database
	EnforceDetection bool   `json:"enforce_detection"`
	ModelName        string `json:"model_name"`       // Face recognition model (e.g. ArcFace)
	DetectorBackend  string `json:"detector_backend"` // Face detector (e.g. retinaface)
	DistanceMetric   string `json:"distance_metric"`  // Distance metric (e.g. cosine)
}

type PersonList []PersonDescriptor

type PersonDescriptor struct {
	ID          string `json:"id"`
	Name        string `json:"name"`
	Description string `json:"description,omitempty"`
	CreatedAt   string `json:"created_at"`
	UpdatedAt   string `json:"updated_at"`
}

type ConfidenceList map[string]float64
type DistanceList map[string]float64
type HashList map[string]string
type IdentityList map[string]string
type SourceHList map[string]float64
type SourceWList map[string]float64
type SourceXList map[string]float64
type SourceYList map[string]float64
type TargetHList map[string]float64
type TargetWList map[string]float64
type TargetXList map[string]float64
type TargetYList map[string]float64
type ThresholdList map[string]float64

type DeepFaceResponse struct {
	Confidence ConfidenceList `json:"confidence"`
	Distance   DistanceList   `json:"distance"`
	Hash       HashList       `json:"hash"`
	Identity   IdentityList   `json:"identity"`
	SourceH    SourceHList    `json:"source_h"`
	SourceW    SourceWList    `json:"source_w"`
	SourceX    SourceXList    `json:"source_x"`
	SourceY    SourceYList    `json:"source_y"`
	TargetH    TargetHList    `json:"target_h"`
	TargetW    TargetWList    `json:"target_w"`
	TargetX    TargetXList    `json:"target_x"`
	TargetY    TargetYList    `json:"target_y"`
	Threshold  ThresholdList  `json:"threshold"`
}

type DeepFaceError struct {
	Message string `json:"message"`
}

func extractName(path string) string {
	parts := strings.Split(path, "/")
	if len(parts) >= 3 {
		return parts[len(parts)-2]
	}
	return ""
}

func getImageList(client *pixmeClient, pixmeUri string, offset int) (ImageApiResponse, error) {
	resp, err := client.Get(pixmeUri + "/api/images/?offset=" + fmt.Sprint(offset) + "&limit=100")
	if err != nil {
		return ImageApiResponse{}, fmt.Errorf("failed to get image list: %w", err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(resp.Body)
		return ImageApiResponse{}, fmt.Errorf("failed to get image list: %s — %s", resp.Status, strings.TrimSpace(string(body)))
	}

	var imageResponse ImageApiResponse
	responseBody, err := io.ReadAll(resp.Body)
	if err != nil {
		return ImageApiResponse{}, fmt.Errorf("failed to read response body: %w", err)
	}
	if err := json.Unmarshal(responseBody, &imageResponse); err != nil {
		return ImageApiResponse{}, fmt.Errorf("failed to unmarshal response body: %w", err)
	}
	return imageResponse, nil
}

func getImageListWithRetry(client *pixmeClient, pixmeUri string, offset int, retries int, delay time.Duration) (ImageApiResponse, error) {
	var lastErr error
	for attempt := 1; attempt <= retries; attempt++ {
		resp, err := getImageList(client, pixmeUri, offset)
		if err == nil {
			return resp, nil
		}
		lastErr = err
		fmt.Printf("Attempt %d/%d failed: %v\n", attempt, retries, err)
		if attempt < retries {
			fmt.Printf("Retrying in %v...\n", delay)
			time.Sleep(delay)
		}
	}
	return ImageApiResponse{}, lastErr
}

func isRetriableStatus(status int) bool {
	return status == http.StatusTooManyRequests || status == http.StatusRequestTimeout || status >= http.StatusInternalServerError
}

func postDeepFaceWithRetry(client *http.Client, url string, payload []byte, retries int, delay time.Duration) (*http.Response, error) {
	var lastErr error
	for attempt := 1; attempt <= retries; attempt++ {
		req, err := http.NewRequest(http.MethodPost, url, bytes.NewReader(payload))
		if err != nil {
			return nil, err
		}
		req.Header.Set("Content-Type", "application/json")
		resp, err := client.Do(req)
		if err == nil {
			if isRetriableStatus(resp.StatusCode) {
				body, _ := io.ReadAll(resp.Body)
				resp.Body.Close()
				lastErr = fmt.Errorf("deepface %s — %s", resp.Status, strings.TrimSpace(string(body)))
			} else {
				return resp, nil
			}
		} else {
			lastErr = err
		}
		if attempt < retries {
			fmt.Printf("DeepFace retry %d/%d after error: %v\n", attempt, retries, lastErr)
			time.Sleep(delay)
		}
	}
	return nil, lastErr
}

func main() {

	fmt.Printf("Starting pixme-facerecognition version %s\n", BuildVersion)

	// Initialize OpenTelemetry tracing
	ctx := context.Background()
	tp, err := initTracer(ctx, "pixme-facerecognition")
	if err != nil {
		fmt.Printf("Warning: Failed to initialize OpenTelemetry tracer: %v\n", err)
	} else {
		defer shutdownTracer(tp)
	}

	// Load configuration from environment variables.
	deepfaceUri := getEnvWithDefault("DEEPFACE_URI", "http://deepface.pixme.svc.cluster.local:5000")
	pixmeUri := getEnvWithDefault("PIXME_URI", "http://pixme.pixme.svc.cluster.local:8080")
	imageBaseUri := getEnvWithDefault("IMAGE_BASE_URI", "http://pixme-static.pixme.svc.cluster.local:80")
	internalAPIKey := getEnvWithDefault("INTERNAL_API_KEY", "")
	modelName := getEnvWithDefault("DEEPFACE_MODEL", "ArcFace")
	detectorBackend := getEnvWithDefault("DEEPFACE_DETECTOR", "retinaface")
	distanceMetric := getEnvWithDefault("DEEPFACE_DISTANCE_METRIC", "cosine")
	maxDistance := getEnvFloat("DEEPFACE_MAX_DISTANCE", 0.40)

	client := &pixmeClient{
		client: &http.Client{
			Timeout:   30 * time.Second,
			Transport: otelhttp.NewTransport(http.DefaultTransport),
		},
		apiKey: internalAPIKey,
	}
	deepfaceTimeout := time.Duration(getEnvInt("DEEPFACE_TIMEOUT_SECONDS", 120)) * time.Second
	deepfaceRetries := getEnvInt("DEEPFACE_RETRIES", 5)
	deepfaceRetryDelay := time.Duration(getEnvInt("DEEPFACE_RETRY_DELAY_SECONDS", 5)) * time.Second
	pixmeRetries := getEnvInt("PIXME_RETRIES", 5)
	pixmeRetryDelay := time.Duration(getEnvInt("PIXME_RETRY_DELAY_SECONDS", 2)) * time.Second
	deepfaceClient := &http.Client{
		Timeout:   deepfaceTimeout,
		Transport: otelhttp.NewTransport(http.DefaultTransport),
	}

	fmt.Printf("Using DeepFace API at %s\n", deepfaceUri)
	fmt.Printf("Using Pixme API at %s\n", pixmeUri)
	fmt.Printf("Using Image Base URI at %s\n", imageBaseUri)
	fmt.Printf("DeepFace model: %s, detector: %s, distance metric: %s, max distance: %.2f\n", modelName, detectorBackend, distanceMetric, maxDistance)
	if internalAPIKey != "" {
		fmt.Println("Internal API key configured")
	}

	imageResponse, err := getImageListWithRetry(client, pixmeUri, 0, pixmeRetries, pixmeRetryDelay)
	if err != nil {
		log.Fatalf("Failed to get initial image list after %d attempts: %v", pixmeRetries, err)
	}

	offset := 0

	for offset < imageResponse.Total {
		imageResponse, err := getImageListWithRetry(client, pixmeUri, offset, pixmeRetries, pixmeRetryDelay)
		if err != nil {
			log.Fatalf("Failed to get image list at offset %d after %d attempts: %v", offset, pixmeRetries, err)
		}

		fmt.Printf("Processing batch at offset %d, count %d, total %d\n", offset, imageResponse.Count, imageResponse.Total)
		count, err := handleImageResponse(imageResponse, client, deepfaceClient, pixmeUri, deepfaceUri, imageBaseUri, modelName, detectorBackend, distanceMetric, maxDistance, deepfaceRetries, deepfaceRetryDelay)
		if err != nil {
			log.Fatalf("Error handling image response at offset %d: %v", offset, err)
		}
		offset += count
	}
	fmt.Println("Face recognition job completed successfully")
}

func handleImageResponse(imageResponse ImageApiResponse, client *pixmeClient, deepfaceClient *http.Client, pixmeUri string, deepfaceUri string, imageBaseUri string, modelName string, detectorBackend string, distanceMetric string, maxDistance float64, deepfaceRetries int, deepfaceRetryDelay time.Duration) (int, error) {
	if imageResponse.Count != len(imageResponse.Images) {
		fmt.Printf("Warning: Count %d does not match number of images %d\n", imageResponse.Count, len(imageResponse.Images))
		return 0, fmt.Errorf("count does not match number of images")
	}
	for _, image := range imageResponse.Images {
		if image.URI == "" {
			fmt.Printf("Warning: Image %s has no URI\n", image.Name)
			continue
		}
		// image.URI comes URL-encoded from the API (e.g. %2Fimages%2F...) — decode it
		decodedURI, err := url.PathUnescape(image.URI)
		if err != nil {
			fmt.Printf("Warning: Failed to decode URI for image %s: %v\n", image.Name, err)
			decodedURI = image.URI // fallback to raw value
		}
		deepFaceRequest := DeepFaceRequest{
			Img:              imageBaseUri + decodedURI,
			DbPath:           "/mnt/faces",
			EnforceDetection: true,
			ModelName:        modelName,
			DetectorBackend:  detectorBackend,
			DistanceMetric:   distanceMetric,
		}

		jsonData, err := json.Marshal(deepFaceRequest)
		if err != nil {
			return 0, fmt.Errorf("failed to marshal DeepFace API request: %s", err)
		}

		fmt.Printf("Sending request to DeepFace API for image %s uri %s %+v\n", image.Name, deepfaceUri+"/find", deepFaceRequest)
		resp, err := postDeepFaceWithRetry(deepfaceClient, deepfaceUri+"/find", jsonData, deepfaceRetries, deepfaceRetryDelay)
		if err != nil {
			return 0, fmt.Errorf("failed to send request to DeepFace API: %s", err)
		}
		defer resp.Body.Close()

		if resp.StatusCode != http.StatusOK {
			body, _ := io.ReadAll(resp.Body)
			if resp.StatusCode == http.StatusBadRequest {
				// With enforce_detection=true, DeepFace returns 400 when no face
				// is detected. This is expected for images without people.
				fmt.Printf("No face detected in image %s, skipping\n", image.Name)
			} else {
				fmt.Printf("DeepFace API error for image %s: %s — %s\n", image.Name, resp.Status, string(body))
			}
			// Skip this image rather than aborting the whole batch
			continue
		}

		// Do something with the DeepFace API response
		var deepFaceResponse DeepFaceResponse
		if err := json.NewDecoder(resp.Body).Decode(&deepFaceResponse); err != nil {
			return 0, fmt.Errorf("failed to decode DeepFace API response: %s", err)
		}

		// Track unique people found in this image to avoid duplicate associations.
		seen := make(map[string]bool)

		// Iterate ALL identity matches, filtering by distance and confidence.
		for key, identity := range deepFaceResponse.Identity {
			distance, hasDistance := deepFaceResponse.Distance[key]

			// Apply strict max distance threshold (overrides DeepFace's default).
			if hasDistance && distance > maxDistance {
				fmt.Printf("Skipping match for image %s: identity=%s distance=%.4f exceeds max_distance=%.4f\n",
					image.Name, identity, distance, maxDistance)
				continue
			}

			// Filter on face confidence if available — reject low-confidence detections
			// which are often false face detections on non-face image regions.
			if confidence, hasConf := deepFaceResponse.Confidence[key]; hasConf && confidence < 0.90 {
				fmt.Printf("Skipping low-confidence detection for image %s: identity=%s confidence=%.4f\n",
					image.Name, identity, confidence)
				continue
			}

			name := extractName(identity)
			if name == "" {
				fmt.Printf("Failed to extract name from path %s\n", identity)
				continue
			}

			// Skip duplicate people within the same image.
			if seen[name] {
				continue
			}
			seen[name] = true

			if hasDistance {
				fmt.Printf("Matched identity %s for image %s (distance=%.4f, max=%.4f)\n",
					name, image.Name, distance, maxDistance)
			} else {
				fmt.Printf("Matched identity %s for image %s\n", name, image.Name)
			}

			if err := associatePersonWithImage(client, pixmeUri, image, name); err != nil {
				fmt.Printf("Error associating %s with image %s: %v\n", name, image.Name, err)
				// Continue processing other matches rather than aborting.
				continue
			}
			fmt.Printf("Successfully associated identity %s with image %s\n", name, image.Name)
		}
	}
	return imageResponse.Count, nil
}

// associatePersonWithImage calls the pixme API to link a person to an image
// and ensures the person entity exists.
func associatePersonWithImage(client *pixmeClient, pixmeUri string, image ImageDescriptor, name string) error {
	assocResp, err := client.Post(pixmeUri+"/api/images/"+image.ID+"/people/"+name, "application/json", nil)
	if err != nil {
		return fmt.Errorf("failed to associate identity: %w", err)
	}
	assocResp.Body.Close()
	if assocResp.StatusCode != http.StatusNoContent && assocResp.StatusCode != http.StatusOK && assocResp.StatusCode != http.StatusConflict {
		return fmt.Errorf("unexpected status associating identity: %s", assocResp.Status)
	}

	type PersonCreateRequest struct {
		Name string `json:"name"`
	}
	personData, err := json.Marshal(PersonCreateRequest{Name: name})
	if err != nil {
		return fmt.Errorf("failed to marshal Person API request: %w", err)
	}
	personResp, err := client.Post(pixmeUri+"/api/people/", "application/json", io.NopCloser(bytes.NewReader(personData)))
	if err != nil {
		return fmt.Errorf("failed to create person: %w", err)
	}
	personResp.Body.Close()
	if personResp.StatusCode != http.StatusCreated && personResp.StatusCode != http.StatusConflict && personResp.StatusCode != http.StatusOK {
		return fmt.Errorf("unexpected status creating person: %s", personResp.Status)
	}
	return nil
}
