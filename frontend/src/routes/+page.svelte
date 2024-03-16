<!-- src/routes/Webcam.svelte -->
<script lang="ts">
    import { Button } from "$lib/components/ui/button";
    import { toast } from "svelte-sonner";
    import * as Card from "$lib/components/ui/card";
  
    let videoStream: MediaStream;
    let videoElement: HTMLVideoElement;
    let canvasElement: HTMLCanvasElement;
    let ctx: CanvasRenderingContext2D;
    let isWebcamOn = false;
    let detectionResults: any[] = [];
  
    async function openWebcam() {
      try {
        videoStream = await navigator.mediaDevices.getUserMedia({ video: true });
        videoElement.srcObject = videoStream;
        videoElement.play();
        isWebcamOn = true;
        sendFrameForInference();
      } catch (error) {
        console.error('Error accessing webcam:', error);
      }
    }
  
    function stopWebcam() {
      if (videoStream) {
        videoStream.getTracks().forEach(track => track.stop());
        videoElement.srcObject = null;
        isWebcamOn = false;
        detectionResults = [];
      }
    }
  
    function handleVideoLoaded(event: Event) {
      videoElement = event.target as HTMLVideoElement;
      canvasElement = document.createElement('canvas');
      canvasElement.width = videoElement.videoWidth;
      canvasElement.height = videoElement.videoHeight;
      ctx = canvasElement.getContext('2d') as CanvasRenderingContext2D;
    }
  
    async function sendFrameForInference() {
      if (videoElement && canvasElement && ctx) {
        ctx.drawImage(videoElement, 0, 0, canvasElement.width, canvasElement.height);
        const imageData = canvasElement.toDataURL('image/jpeg');
  
        try {
          const response = await fetch('http://localhost:8000/detect', {
            method: 'POST',
            headers: {
              'Content-Type': 'application/json',
            },
            body: JSON.stringify({ image: imageData }),
          });
  
          const results = await response.json();
          detectionResults = results;
        } catch (error) {
          console.error('Error during inference:', error);
        }
      }
  
      if (isWebcamOn) {
        requestAnimationFrame(sendFrameForInference);
      }
    }
  </script>
  
  <div class="flex flex-col items-center justify-center min-h-screen">
    <div class="mb-4 relative w-full max-w-xl">
        <h1 class="text-4xl font-bold text-center p-2">Uno Card Detection</h1>
        <p class="text-2xl text-center p-6">Turn on your webcam and show your UNO Card</p>
      {#if isWebcamOn}
        <video class="w-full" bind:this={videoElement} autoplay on:loadedmetadata={handleVideoLoaded}>
          <track kind="captions" src="captions.vtt" srclang="en" label="English" default />
        </video>
      {:else}
        <Card.Root class="w-full">
          <Card.Content class="flex aspect-video items-center justify-center p-6">
            <span class="text-4xl font-semibold">Webcam Off</span>
          </Card.Content>
        </Card.Root>
      {/if}
    </div>
    <div class="space-x-4 mb-4">
      <Button on:click={() => {
        isWebcamOn = true;
        openWebcam();
        toast("Webcam On", { description: "Webcam is turned on" });
      }}>
        On Webcam
      </Button>
      <Button on:click={() => {
        isWebcamOn = false;
        stopWebcam();
        toast("Webcam Off", { description: "Webcam is turned off" });
      }}>
        Stop Webcam
      </Button>
    </div>
    {#if detectionResults.length > 0}
      <div class="space-y-2">
        {#each detectionResults as result}
          <h2 class="text-xl font-bold">Card {result.class_name} - {result.score.toFixed(2)*100}%</h2>
        {/each}
      </div>
    {:else}
      <h2 class="text-xl font-bold">No detection</h2>
    {/if}
  </div>