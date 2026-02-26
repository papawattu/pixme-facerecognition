package main

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"net/http"
	"net/url"
	"os"
	"strings"
	"time"
)

func getEnvWithDefault(key, defaultValue string) string {
	value := os.Getenv(key)
	if value == "" {
		return defaultValue
	}
	return value
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

func main() {

	// Load configuration from environment variables.
	deepfaceUri := getEnvWithDefault("DEEPFACE_URI", "http://deepface.pixme.svc.cluster.local:5000")
	pixmeUri := getEnvWithDefault("PIXME_URI", "http://pixme.pixme.svc.cluster.local:8080")
	imageBaseUri := getEnvWithDefault("IMAGE_BASE_URI", "http://pixme-static.pixme.svc.cluster.local:80")
	internalAPIKey := getEnvWithDefault("INTERNAL_API_KEY", "")

	client := &pixmeClient{
		client: &http.Client{Timeout: 30 * time.Second},
		apiKey: internalAPIKey,
	}

	fmt.Printf("Using DeepFace API at %s\n", deepfaceUri)
	fmt.Printf("Using Pixme API at %s\n", pixmeUri)
	fmt.Printf("Using Image Base URI at %s\n", imageBaseUri)
	if internalAPIKey != "" {
		fmt.Println("Internal API key configured")
	}

	// Retry initial connection with backoff — pod networking may not be ready immediately
	var imageResponse ImageApiResponse
	var err error
	for attempt := 1; attempt <= 5; attempt++ {
		imageResponse, err = getImageList(client, pixmeUri, 0)
		if err == nil {
			break
		}
		fmt.Printf("Attempt %d/5 failed: %v\n", attempt, err)
		if attempt < 5 {
			delay := time.Duration(attempt) * 2 * time.Second
			fmt.Printf("Retrying in %v...\n", delay)
			time.Sleep(delay)
		}
	}
	if err != nil {
		log.Fatalf("Failed to get initial image list after 5 attempts: %v", err)
	}

	offset := 0

	for offset < imageResponse.Total {
		imageResponse, err := getImageList(client, pixmeUri, offset)
		if err != nil {
			log.Fatalf("Failed to get image list at offset %d: %v", offset, err)
		}

		fmt.Printf("Processing batch at offset %d, count %d, total %d\n", offset, imageResponse.Count, imageResponse.Total)
		count, err := handleImageResponse(imageResponse, client, pixmeUri, deepfaceUri, imageBaseUri)
		if err != nil {
			log.Fatalf("Error handling image response at offset %d: %v", offset, err)
		}
		offset += count
	}
	fmt.Println("Face recognition job completed successfully")
}

func handleImageResponse(imageResponse ImageApiResponse, client *pixmeClient, pixmeUri string, deepfaceUri string, imageBaseUri string) (int, error) {
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
			EnforceDetection: false,
		}

		jsonData, err := json.Marshal(deepFaceRequest)
		if err != nil {
			return 0, fmt.Errorf("failed to marshal DeepFace API request: %s", err)
		}

		fmt.Printf("Sending request to DeepFace API for image %s uri %s %+v\n", image.Name, deepfaceUri+"/find", deepFaceRequest)
		resp, err := http.Post(deepfaceUri+"/find", "application/json", io.NopCloser(bytes.NewReader(jsonData)))
		if err != nil {
			return 0, fmt.Errorf("failed to send request to DeepFace API: %s", err)
		}
		defer resp.Body.Close()

		if resp.StatusCode != http.StatusOK {
			body, _ := io.ReadAll(resp.Body)
			fmt.Printf("DeepFace API error for image %s: %s — %s\n", image.Name, resp.Status, string(body))
			// Skip this image rather than aborting the whole batch
			continue
		}

		// Do something with the DeepFace API response
		var deepFaceResponse DeepFaceResponse
		if err := json.NewDecoder(resp.Body).Decode(&deepFaceResponse); err != nil {
			return 0, fmt.Errorf("failed to decode DeepFace API response: %s", err)
		}
		if len(deepFaceResponse.Identity) > 0 {
			path := deepFaceResponse.Identity["0"]
			name := extractName(path)
			if name == "" {
				fmt.Printf("Failed to extract name from path %s\n", path)
				return 0, fmt.Errorf("failed to extract name from path %s", path)
			}

			fmt.Printf("Associating identity %s with image %s\n", name, image.Name)
			assocResp, err := client.Post(pixmeUri+"/api/images/"+image.ID+"/people/"+name, "application/json", nil)
			if err != nil {
				fmt.Printf("Failed to associate identity with image %s: %s\n", image.Name, err)
				return 0, fmt.Errorf("failed to associate identity with image %s: %s", image.Name, err)
			}
			assocResp.Body.Close()
			if assocResp.StatusCode != http.StatusNoContent && assocResp.StatusCode != http.StatusOK && assocResp.StatusCode != http.StatusConflict {
				fmt.Printf("Failed to associate identity with image %s: %s\n", image.Name, assocResp.Status)
				return 0, fmt.Errorf("failed to associate identity with image %s: %s", image.Name, assocResp.Status)
			}

			type PersonCreateRequest struct {
				Name string `json:"name"`
			}
			personData, err := json.Marshal(PersonCreateRequest{Name: name})
			if err != nil {
				return 0, fmt.Errorf("failed to marshal Person API request: %s", err)
			}
			personResp, err := client.Post(pixmeUri+"/api/people/", "application/json", io.NopCloser(bytes.NewReader(personData)))
			if err != nil {
				fmt.Printf("Failed to create person %s: %s\n", name, err)
				return 0, fmt.Errorf("failed to create person %s: %s", name, err)
			}
			personResp.Body.Close()
			if personResp.StatusCode != http.StatusCreated && personResp.StatusCode != http.StatusConflict && personResp.StatusCode != http.StatusOK {
				fmt.Printf("Failed to create person %s: %s\n", name, personResp.Status)
				return 0, fmt.Errorf("failed to create person %s: %s", name, personResp.Status)
			}
			fmt.Printf("Successfully associated identity %s with image %s\n", name, image.Name)
		}
	}
	return imageResponse.Count, nil
}
