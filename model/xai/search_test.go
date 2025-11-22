package xai

import (
	"testing"
	"time"

	pb "github.com/zchee/tumix/model/xai/pb/xai/api/v1"
)

func TestSearchParametersProto(t *testing.T) {
	from := time.Date(2024, 1, 2, 3, 4, 5, 0, time.UTC)
	params := SearchParameters{
		Sources:          []*pb.Source{WebSource("US", []string{"example.com"}, nil, true)},
		Mode:             SearchModeOn,
		FromDate:         &from,
		ReturnCitations:  true,
		MaxSearchResults: 7,
	}
	proto := params.Proto()
	if proto.GetMode() != pb.SearchMode_ON_SEARCH_MODE {
		t.Fatalf("mode not converted")
	}
	if proto.FromDate == nil || !proto.FromDate.AsTime().Equal(from) {
		t.Fatalf("from date mismatch")
	}
	if proto.GetMaxSearchResults() != 7 {
		t.Fatalf("max search results mismatch")
	}
	if len(proto.Sources) != 1 || proto.Sources[0].GetWeb() == nil {
		t.Fatalf("web source missing")
	}
}
