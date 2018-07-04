package storageScanner

import (
	"bytes"
	"context"
	"log"
	"os"
	"strings"
	"testing"

	"github.com/presidium-io/stow"
	"github.com/stretchr/testify/mock"

	"github.com/stretchr/testify/assert"

	message_types "github.com/presidium-io/presidium-genproto/golang"
	"github.com/presidium-io/presidium/pkg/cache"
	cache_mock "github.com/presidium-io/presidium/pkg/cache/mock"
	analyzer "github.com/presidium-io/presidium/pkg/modules/analyzer"
	"github.com/presidium-io/presidium/pkg/storage"
)

var (
	// Azure emulator connection string
	storageName = "devstoreaccount1"
	storageKey  = "Eby8vdM02xNOcqFlqUwJPLlmEtlCDXJ1OUzFT50uSRZ6IFsuFq2UVErCz4I6tq/K1SZFPTOtr/KBHBeksoGMGw=="
)

type MyMockedObject struct {
	mock.Mock
}

func (m *MyMockedObject) InvokeAnalyze(c context.Context, analyzeRequest *message_types.AnalyzeRequest, text string) (*message_types.AnalyzeResponse, error) {
	args := m.Mock.Called()
	y := args.Get(0).(*message_types.AnalyzeResponse)
	return y, args.Error(1)
}

func TestAzureScanAndAnalyze(t *testing.T) {
	var testCache = cache_mock.New()
	kind, config := storage.CreateAzureConfig(storageName, storageKey)

	content := "Please call me. My phone number is (555) 253-0000."
	var analyzeService *message_types.AnalyzeServiceClient

	analyzerObj := &MyMockedObject{}
	var serviceMock analyzer.Analyzer = analyzerObj
	analyzerObj.On("InvokeAnalyze", mock.Anything, mock.Anything, mock.Anything).Return(getAnalyzerMockResult(), nil)

	api, _ := storage.New(testCache, kind, config, analyzeService)

	container, err := api.CreateContainer("test")
	if err != nil {
		log.Fatal(err.Error())
	}

	// Add data to
	md1 := map[string]interface{}{"stowmetadata": "foo"}
	container.Put("file1", strings.NewReader(content), int64(len(content)), md1)

	// Set log output
	var buf bytes.Buffer
	log.SetOutput(&buf)
	defer func() {
		log.SetOutput(os.Stderr)
	}()

	scanAndAnalyaze(container, testCache, serviceMock)
	// validate output
	assert.Contains(t, buf.String(), "Found: \"PHONE_NUMBER\", propability: 1.000000, Location: start:153 end:163 length:10")
	scanAndAnalyaze(container, testCache, serviceMock)
	assert.Contains(t, buf.String(), "Item was already scanned file1")
}

func getAnalyzerMockResult() *message_types.AnalyzeResponse {
	location := &message_types.Location{
		Start: 153, End: 163, Length: 10,
	}
	results := [](*message_types.AnalyzeResult){
		&message_types.AnalyzeResult{
			Field:       message_types.FieldTypes_PHONE_NUMBER,
			Text:        "(555) 253-0000",
			Probability: 1.0,
			Location:    location,
		},
	}
	response := &message_types.AnalyzeResponse{
		AnalyzeResults: results,
	}
	return response
}

func scanAndAnalyaze(container stow.Container, testCache cache.Cache, serviceMock analyzer.Analyzer) {
	var analyzeRequest *message_types.AnalyzeRequest
	err := stow.Walk(container, stow.CursorStart, 100, func(item stow.Item, err error) error {
		if err != nil {
			return err
		}

		ScanAndAnalyze(&testCache, container, item, &serviceMock, analyzeRequest)
		return nil
	})

	if err != nil {
		log.Fatal(err.Error())
		return
	}
}
