package server

import (
	"context"
	"net/http"
	"os"
	"os/signal"
	"strconv"
	"time"

	"github.com/gin-contrib/cors"
	"github.com/gin-contrib/zap"
	"github.com/gin-gonic/gin"
	"golang.org/x/sync/errgroup"

	log "github.com/presid-io/presidio/pkg/logger"
)

var (
	port   int
	g      errgroup.Group
	server *http.Server
)

//Setup ...
func Setup(_port int) *gin.Engine {
	if _port == 0 {
		_port = 8080
	}
	r := gin.New()
	r.Use(gin.Recovery())

	// Add a ginzap middleware, which:
	//   - Logs all requests, like a combined access and error log.
	//   - Logs to stdout.
	//   - RFC3339 with UTC time format.
	r.Use(ginzap.Ginzap(log.GetInstance(), time.RFC3339, true))
	r.Use(cors.Default())
	r.GET("/healthz", healthCheck)
	port = _port
	return r
}

//Start ...
func Start(r *gin.Engine) {
	server = &http.Server{
		Addr:    ":" + strconv.Itoa(port),
		Handler: r,
	}

	g.Go(func() error {
		return server.ListenAndServe()
	})

	if err := g.Wait(); err != nil {
		log.Fatal(err.Error())
	}
}

//Stop ...
func Stop() {
	quit := make(chan os.Signal)
	signal.Notify(quit, os.Interrupt)

	log.Info("Shutdown Server ...")

	ctx, cancel := context.WithTimeout(context.Background(), time.Second)
	defer cancel()
	if err := server.Shutdown(ctx); err != nil {
		log.Fatal(err.Error())
	}
	log.Info("Server exiting")
}
func healthCheck(c *gin.Context) {
	WriteResponse(c, http.StatusOK, gin.H{
		"iam": "alive",
	})

}

//WriteResponse ...
func WriteResponse(
	c *gin.Context,
	statusCode int,
	responseBody interface{},
) {
	c.JSON(statusCode, responseBody)
}
