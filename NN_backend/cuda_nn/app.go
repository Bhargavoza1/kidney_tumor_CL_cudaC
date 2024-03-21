package main

import (
	"encoding/base64"
	"encoding/json"
	"fmt"
	"io/ioutil"
	"net/http"
	"os"
)

/*
#cgo LDFLAGS: -L ./build -lapp -lstdc++
#include "app.h"
*/
import "C"

func main() {
    C.train()
    mux := http.NewServeMux()
    mux.HandleFunc("/api/train", train)
    mux.HandleFunc("/api/test", test)
    mux.HandleFunc("/api/predict", predict)
    fmt.Println("Server started on port 8080")
    http.ListenAndServe(":8080", addCorsHeaders(mux))
}

func test(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		w.WriteHeader(http.StatusMethodNotAllowed)
		fmt.Fprintf(w, "Method Not Allowed")
		return
	}

	fmt.Fprintf(w, "test started\n")

	C.test()

	fmt.Fprintf(w, "test done")
}

func train(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		w.WriteHeader(http.StatusMethodNotAllowed)
		fmt.Fprintf(w, "Method Not Allowed")
		return
	}

	fmt.Fprintf(w, "train started\n")
	C.train()

	fmt.Fprintf(w, "train done")
}

func predict(w http.ResponseWriter, r *http.Request) {
	// Check if the request method is POST
	if r.Method != http.MethodPost {
		w.WriteHeader(http.StatusMethodNotAllowed)
		fmt.Fprintf(w, "Method Not Allowed")
		return
	}

	// Decode JSON data from request body
	var imageData struct {
		Image string `json:"image"`
	}

	// Read request body
	body, err := ioutil.ReadAll(r.Body)
	if err != nil {
		w.WriteHeader(http.StatusInternalServerError)
		fmt.Fprintf(w, "Failed to read request body: %v", err)
		return
	}

	// Parse JSON data
	err = json.Unmarshal(body, &imageData)
	if err != nil {
		w.WriteHeader(http.StatusBadRequest)
		fmt.Fprintf(w, "Failed to parse JSON: %v", err)
		return
	}

	// Decode base64 image data
	decoded, err := base64.StdEncoding.DecodeString(imageData.Image)
	if err != nil {
		w.WriteHeader(http.StatusBadRequest)
		fmt.Fprintf(w, "Failed to decode base64 image: %v", err)
		return
	}

	// Write the decoded image data to a file
	outFile, err := os.Create("kidny.jpg") // Change the file extension based on the image type
	if err != nil {
		w.WriteHeader(http.StatusInternalServerError)
		fmt.Fprintf(w, "Failed to create file: %v", err)
		return
	}
	defer outFile.Close()

	_, err = outFile.Write(decoded)
	if err != nil {
		w.WriteHeader(http.StatusInternalServerError)
		fmt.Fprintf(w, "Failed to write to file: %v", err)
		return
	}

	//fmt.Fprintf(w, "Image uploaded successfully\n")

	result := int(C.predict())
	if result == 1 {
		fmt.Fprintf(w, "Tumor")
	} else {
		fmt.Fprintf(w, "Normal")
	}
}

// Middleware function to add CORS headers
func addCorsHeaders(handler http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		// Set CORS headers
		w.Header().Set("Access-Control-Allow-Origin", "*")
		w.Header().Set("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
		w.Header().Set("Access-Control-Allow-Headers", "Content-Type")

		// Handle preflight requests
		if r.Method == "OPTIONS" {
			w.WriteHeader(http.StatusOK)
			return
		}

		// Continue with the next handler
		handler.ServeHTTP(w, r)
	})
}
