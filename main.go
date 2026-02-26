package main

import (
	"bytes"
	"database/sql"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os"
	"strings"

	"github.com/joho/godotenv"
	_ "github.com/lib/pq" // PostgreSQL driver
)

var pool *sql.DB // Database connection pool.

func getEnvWithDefault(key, defaultValue string) string {
	value := os.Getenv(key)
	if value == "" {
		return defaultValue
	}
	return value
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

func getImageList(pixmeUri string, offset int) (ImageApiResponse, error) {
	resp, err := http.Get(pixmeUri + "/api/images/?offset=" + fmt.Sprint(offset) + "&limit=100")
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
	// load from .env file if it exists
	_ = godotenv.Load()
	deepfaceUri := getEnvWithDefault("DEEPFACE_URI", "http://deepface.pixme.svc.cluster.local:5000")
	pixmeUri := getEnvWithDefault("PIXME_URI", "http://pixme.pixme.svc.cluster.local:8080")

	fmt.Printf("Using DeepFace API at %s\n", deepfaceUri)
	fmt.Printf("Using Pixme API at %s\n", pixmeUri)
	// Fetch the initial image list.
	imageResponse, err := getImageList(pixmeUri, 0)
	if err != nil {
		panic(err)
	}

	offset := 0

	for offset < imageResponse.Total {
		imageResponse, err := getImageList(pixmeUri, offset)
		if err != nil {
			panic(err)
		}

		fmt.Printf("Image %+v\n", imageResponse)
		count, err := handleImageResponse(imageResponse, pixmeUri, deepfaceUri)
		if err != nil {
			fmt.Printf("Error handling image response: %s\n", err)
			panic(err)
		}
		offset += count
	}
}

func handleImageResponse(imageResponse ImageApiResponse, pixmeUri string, deepfaceUri string) (int, error) {
	if imageResponse.Count != len(imageResponse.Images) {
		fmt.Printf("Warning: Count %d does not match number of images %d\n", imageResponse.Count, len(imageResponse.Images))
		return 0, fmt.Errorf("count does not match number of images")
	}
	for _, image := range imageResponse.Images {
		if image.URI == "" {
			fmt.Printf("Warning: Image %s has no URI\n", image.Name)
			continue
		}
		//fmt.Printf("Processing image: %s\n", image.Name)
		deepFaceRequest := DeepFaceRequest{
			Img:              pixmeUri + image.URI,
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

		// Do something with the DeepFace API response
		var deepFaceResponse DeepFaceResponse
		if err := json.NewDecoder(resp.Body).Decode(&deepFaceResponse); err != nil {
			return 0, fmt.Errorf("failed to decode DeepFace API response: %s", err)
		}
		if resp.StatusCode != http.StatusOK {
			fmt.Printf("DeepFace API error: %s\n", resp.Status)
			return 0, fmt.Errorf("deepface API error: %s", resp.Status)
		} else {
			if len(deepFaceResponse.Identity) > 0 {
				path := deepFaceResponse.Identity["0"]
				fmt.Printf("Associating identity %s with image %s\n", extractName(path), image.Name)
				resp, err := http.Post(pixmeUri+"/api/images/"+image.ID+"/people/"+extractName(path), "application/json", nil)

				if err != nil {
					fmt.Printf("Failed to associate identity with image %s: %s\n", image.Name, err)
					return 0, fmt.Errorf("failed to associate identity with image %s: %s", image.Name, err)
				} else {
					defer resp.Body.Close()
					if resp.StatusCode != http.StatusNoContent && resp.StatusCode != http.StatusOK && resp.StatusCode != http.StatusConflict {
						fmt.Printf("Failed to associate identity with image %s: %s\n", image.Name, resp.Status)
						return 0, fmt.Errorf("failed to associate identity with image %s: %s", image.Name, resp.Status)
					} else {
						name := extractName(path)
						if name == "" {
							fmt.Printf("Failed to extract name from path %s\n", path)
							return 0, fmt.Errorf("failed to extract name from path %s", path)
						}
						type PersonCreateRequest struct {
							Name string `json:"name"`
						}
						personCreateRequest := PersonCreateRequest{
							Name: name,
						}
						jsonData, err := json.Marshal(personCreateRequest)
						if err != nil {
							return 0, fmt.Errorf("failed to marshal Person API request: %s", err)
						}
						resp, err := http.Post(pixmeUri+"/api/people/", "application/json", io.NopCloser(bytes.NewReader(jsonData)))
						if err != nil {
							fmt.Printf("Failed to create person %s: %s\n", name, err)
							return 0, fmt.Errorf("failed to create person %s: %s", name, err)
						} else {
							defer resp.Body.Close()
							if resp.StatusCode != http.StatusCreated && resp.StatusCode != http.StatusConflict && resp.StatusCode != http.StatusOK {
								fmt.Printf("Failed to create person %s: %s\n", name, resp.Status)
								return 0, fmt.Errorf("failed to create person %s: %s", name, resp.Status)
							} else {
								fmt.Printf("Successfully associated identity %s with image %s\n", name, image.Name)
							}
						}
					}
				}
			}
		}
	}
	return imageResponse.Count, nil
}
