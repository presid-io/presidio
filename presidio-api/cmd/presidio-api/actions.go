package main

import (
	"fmt"
	"net/http"

	"github.com/gin-gonic/gin"

	"github.com/presid-io/presidio/pkg/service-discovery/consul"

	log "github.com/presid-io/presidio/pkg/logger"
	analyzer "github.com/presid-io/presidio/pkg/modules/analyzer"
	"github.com/presid-io/presidio/pkg/rpc"

	message_types "github.com/presid-io/presidio-genproto/golang"
	helper "github.com/presid-io/presidio/pkg/helper"
	server "github.com/presid-io/presidio/pkg/server"
	templates "github.com/presid-io/presidio/pkg/templates"
)

var analyzeService *message_types.AnalyzeServiceClient
var anonymizeService *message_types.AnonymizeServiceClient

func setupGrpcServices() {
	store := consul.New()
	var err error
	var analyzerSvcHost, anonymizerSvcHost string

	analyzerSvcHost, err = store.GetService("analyzer")
	if err != nil {
		log.Fatal(fmt.Sprintf("analyzer service address is empty %q", err))
	}

	anonymizerSvcHost, err = store.GetService("anonymizer")
	if err != nil {
		log.Fatal(fmt.Sprintf("anonymizer service address is empty %q", err))
	}

	analyzeService, err = rpc.SetupAnalyzerService(analyzerSvcHost)
	if err != nil {
		log.Error(fmt.Sprintf("Connection to analyzer service failed %q", err))
	}
	anonymizeService, err = rpc.SetupAnonymizeService(anonymizerSvcHost)
	if err != nil {
		log.Error(fmt.Sprintf("Connection to anonymizer service failed %q", err))
	}

}

func (api *API) analyze(c *gin.Context) {
	var analyzeAPIRequest message_types.AnalyzeApiRequest

	if c.Bind(&analyzeAPIRequest) == nil {
		id := analyzeAPIRequest.AnalyzeTemplateId
		project := c.Param("project")
		res := api.invokeAnalyze(project, id, analyzeAPIRequest.Text, c)
		if res == nil {
			return
		}
		server.WriteResponse(c, http.StatusOK, res)
	}
}

func (api *API) anonymize(c *gin.Context) {
	var anonymizeAPIRequest message_types.AnonymizeApiRequest

	if c.Bind(&anonymizeAPIRequest) == nil {

		id := anonymizeAPIRequest.AnalyzeTemplateId
		project := c.Param("project")
		analyzeRes := api.invokeAnalyze(project, id, anonymizeAPIRequest.Text, c)
		if analyzeRes == nil {
			return
		}

		id = anonymizeAPIRequest.AnonymizeTemplateId
		anonymizeRes := api.invokeAnonymize(project, id, anonymizeAPIRequest.Text, analyzeRes.AnalyzeResults, c)
		if anonymizeRes == nil {
			return
		}
		server.WriteResponse(c, http.StatusOK, anonymizeRes)
	}
}

func (api *API) invokeAnonymize(project string, id string, text string, results []*message_types.AnalyzeResult, c *gin.Context) *message_types.AnonymizeResponse {
	anonymizeKey := templates.CreateKey(project, "anonymize", id)
	result, err := api.templates.GetTemplate(anonymizeKey)

	if err != nil {
		server.WriteResponse(c, http.StatusBadRequest, fmt.Sprintf("Failed to retrieve template %q", err))
		return nil
	}
	srv := *anonymizeService
	anonymizeTemplate := &message_types.AnonymizeTemplate{}
	err = helper.ConvertJSONToInterface(result, anonymizeTemplate)
	if err != nil {
		server.WriteResponse(c, http.StatusBadRequest, fmt.Sprintf("Failed to convert template %q", err))
		return nil
	}
	request := &message_types.AnonymizeRequest{
		Template:       anonymizeTemplate,
		Text:           text,
		AnalyzeResults: results,
	}
	res, err := srv.Apply(c, request)
	if err != nil {
		server.WriteResponse(c, http.StatusBadRequest, err.Error())
		return nil
	}
	return res
}

func (api *API) invokeAnalyze(project string, id string, text string, c *gin.Context) *message_types.AnalyzeResponse {
	analyzeKey := templates.CreateKey(project, "analyze", id)
	analyzeRequest, err := analyzer.GetAnalyzeRequest(api.templates, analyzeKey)

	if err != nil {
		server.WriteResponse(c, http.StatusBadRequest, err.Error())
		return nil
	}

	analyzerObj := analyzer.New(analyzeService)
	analyzeResponse, err := analyzerObj.InvokeAnalyze(c, analyzeRequest, text)
	if err != nil {
		server.WriteResponse(c, http.StatusBadRequest, err.Error())
		return nil
	}
	return analyzeResponse
}
