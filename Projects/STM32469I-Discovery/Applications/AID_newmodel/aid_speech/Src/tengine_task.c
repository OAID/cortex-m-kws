#include <stdio.h>
#include "tengine_task.h"

extern int tprintf(const char * str, ...);

static const struct tiny_graph* tiny_graph;
extern const struct tiny_graph* get_tiny_graph(void);
extern void free_tiny_graph(const struct tiny_graph*);

/* Private functions ---------------------------------------------------------*/
static void log_func(const char* info)
{
    printf("%s", info);
}

graph_t tengine_lite_init(graph_t graph)
{
    // Step 0, init tengine
    init_tengine();	

    set_log_output(log_func);
    
    // step 1, get the model structure data
    tiny_graph = get_tiny_graph();

    // step 2, create the graph
    graph = create_graph(NULL, "tiny", ( void* )tiny_graph);
    if(graph == NULL)
    {
        printf("create graph from tiny model failed\n");
        goto TENGINE_ERR;
    }

    // step 3, prerun graph
    if(prerun_graph(graph) < 0)
    {
        printf("prerun graph failed\n");
        goto TENGINE_ERR;
    }		
    
    return graph;

TENGINE_ERR:
    return NULL;
}


void tengine_lite_release(graph_t graph)
{
    postrun_graph(graph);
    destroy_graph(graph);
    free_tiny_graph(tiny_graph);
    release_tengine();
}